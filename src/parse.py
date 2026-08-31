"""
XML -> tabular.

Reads the three raw DFL XML files per match (matchinformation, events_raw,
positions_raw_observed) and writes:
    data/processed/matches.csv
    data/processed/players.csv
    data/processed/events.csv
    data/processed/positions/<match_id>.parquet
    data/processed/positions_sample.csv   (first 2000 rows of one match)
"""

import math
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from lxml import etree

# --- config -----------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    path = path or os.path.join(REPO_ROOT, "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)

CONFIG = load_config()
RAW_DIR = os.path.join(REPO_ROOT, CONFIG["paths"]["raw_data"])
OUT_DIR = os.path.join(REPO_ROOT, CONFIG["paths"]["processed_data"])
MATCH_IDS = CONFIG["matches"]
TARGET_HZ = CONFIG["target_frame"]
SOURCE_HZ = 25
KEEP_EVERY = int(SOURCE_HZ / TARGET_HZ + 0.5)
ACTUAL_HZ = SOURCE_HZ / KEEP_EVERY


def tag(elem):
    """Element tag without namespace, e.g. '{ns}Foo' -> 'Foo'."""
    return etree.QName(elem).localname


def find_file(pattern_fragment, match_id):
    """The DFL_02/03/04 filenames carry a DFL-COM-... competition id we
    don't otherwise use, so match on match_id + the fixed fragment instead
    of hardcoding the competition id."""
    for fname in os.listdir(RAW_DIR):
        if pattern_fragment in fname and match_id in fname:
            return os.path.join(RAW_DIR, fname)
    raise FileNotFoundError(f"No file matching '{pattern_fragment}' for {match_id} in {RAW_DIR}")


# -----------------match information --------------------------------------

def parse_match_information(match_id):
    """Returns (match_row: dict, player_rows: list[dict])."""
    path = find_file("matchinformation", match_id)
    root = etree.parse(path).getroot()

    general = root.find(".//General")
    env = root.find(".//Environment")

    match_row = {
        "match_id": match_id,
        "competition_name": general.get("CompetitionName"),
        "matchday": general.get("MatchDay"),
        "season": general.get("Season"),
        "kickoff_time": general.get("KickoffTime"),
        "home_team_id": general.get("HomeTeamId"),
        "home_team_name": general.get("HomeTeamName"),
        "away_team_id": general.get("GuestTeamId"),
        "away_team_name": general.get("GuestTeamName"),
        "result": general.get("Result"),
        "pitch_x": float(env.get("PitchX")) if env is not None else None,
        "pitch_y": float(env.get("PitchY")) if env is not None else None,
    }

    player_rows = []
    for team_elem in root.findall(".//Teams/Team"):
        team_id = team_elem.get("TeamId")
        team_name = team_elem.get("TeamName")
        team_role = team_elem.get("Role")  # 'home' / 'guest' in the raw data
        for p in team_elem.findall(".//Players/Player"):
            player_rows.append({
                "person_id": p.get("PersonId"),
                "match_id": match_id,
                "team_id": team_id,
                "team_name": team_name,
                "team_role": team_role,
                "first_name": p.get("FirstName"),
                "last_name": p.get("LastName"),
                "name": f"{p.get('FirstName', '')} {p.get('LastName', '')}".strip(),
                "shirt_number": p.get("ShirtNumber"),
                "playing_position": p.get("PlayingPosition"),
                "starting": p.get("Starting") == "true",
            })

    return match_row, player_rows


# ------------------- events ---------------------------------------------------


COLUMN_MAP = {
    "Player": "player_id",
    "Team": "team_id_event",
    "Recipient": "recipient_id",
    "Evaluation": "evaluation",
}


def build_chain(event_elem):
    """Walk the single nested-child chain inside an <Event> (KickOff ->
    Play -> Pass, etc). Real data for the 7 matches never branches (verified
    empirically), but we still detect and report a branch rather than
    silently taking child [0] and dropping sibling information."""
    chain = []
    branched = False
    cur = event_elem
    while True:
        children = [c for c in cur if isinstance(c.tag, str)]
        if not children:
            break
        if len(children) > 1:
            branched = True
        chain.append(children[0])
        cur = children[0]
    return chain, branched


def parse_events(match_id):
    """Returns (events_df, n_delete, n_branched)."""
    path = find_file("events_raw", match_id)
    rows = []
    n_delete = 0
    n_branched = 0

    context = etree.iterparse(path, events=("end",), tag="Event")
    for _, elem in context:
        chain, branched = build_chain(elem)
        if branched:
            n_branched += 1

        top_level_type = tag(chain[0]) if chain else None
        event_type_path = "_".join(tag(c) for c in chain)

        if top_level_type == "Delete":
            n_delete += 1
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
            continue

        x_event = elem.get("X-Position")
        y_event = elem.get("Y-Position")
        if x_event is None:
            x_event = elem.get("X-Source-Position")
            y_event = elem.get("Y-Source-Position")
        x_event = float(x_event) if x_event is not None else None
        y_event = float(y_event) if y_event is not None else None

        row = {
            "match_id": match_id,
            "event_id": elem.get("EventId"),
            "timestamp": elem.get("EventTime"),
            "event_type_path": event_type_path,
            "top_level_type": top_level_type,
            "x_event": x_event,
            "y_event": y_event,
            "x_centred": (x_event - 52.5) if x_event is not None else None,
            "y_centred": (y_event - 34.0) if y_event is not None else None,
            "x_source_position": elem.get("X-Source-Position"),
            "y_source_position": elem.get("Y-Source-Position"),
            "x_position": elem.get("X-Position"),
            "y_position": elem.get("Y-Position"),
            "calculated_frame": elem.get("CalculatedFrame"),
            "calculated_timestamp": elem.get("CalculatedTimestamp"),
            "x_position_from_tracking": elem.get("X-PositionFromTracking"),
            "y_position_from_tracking": elem.get("Y-PositionFromTracking"),
        }

        for src_attr, col_name in COLUMN_MAP.items():
            val = None
            for c in chain:
                if src_attr in c.attrib:
                    val = c.get(src_attr)
                    break
            row[col_name] = val

        # {tag}_{attr} prefix so no information is lost even for event
        for c in chain:
            prefix = tag(c).lower()
            for k, v in c.attrib.items():
                row[f"{prefix}_{k.lower()}"] = v

        rows.append(row)

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["team_id"] = df["team_id_event"]
    return df, n_delete, n_branched


POSITION_COLUMNS = [
    "match_id", "game_section", "frame", "timestamp", "person_id",
    "team_id", "x", "y", "z", "d", "s", "a", "minute",
    "ball_possession", "ball_status",
]


def _frameset_to_df(match_id, fs_elem, keep_every):
    game_section = fs_elem.get("GameSection")
    team_id = fs_elem.get("TeamId")
    person_id = fs_elem.get("PersonId") or ("BALL" if team_id == "BALL" else None)
    is_ball = team_id == "BALL"

    kept_rows = []
    total_d_cm = 0.0
    for i, f in enumerate(fs_elem):
        if tag(f) != "Frame":
            continue
        d_val = f.get("D")
        if d_val is not None:
            total_d_cm += float(d_val)
        if i % keep_every != 0:
            continue
        kept_rows.append({
            "match_id": match_id,
            "game_section": game_section,
            "frame": int(f.get("N")),
            "timestamp": f.get("T"),
            "person_id": person_id,
            "team_id": team_id,
            "x": float(f.get("X")),
            "y": float(f.get("Y")),
            "z": float(f.get("Z")) if is_ball and f.get("Z") is not None else None,
            "d": float(d_val) if d_val is not None else None,
            "s": float(f.get("S")) if f.get("S") is not None else None,
            "a": float(f.get("A")) if f.get("A") is not None else None,
            "minute": int(f.get("M")) if f.get("M") is not None else None,
            "ball_possession": int(f.get("BallPossession")) if is_ball and f.get("BallPossession") is not None else None,
            "ball_status": int(f.get("BallStatus")) if is_ball and f.get("BallStatus") is not None else None,
        })

    df = pd.DataFrame(kept_rows, columns=POSITION_COLUMNS)
    return df, person_id, total_d_cm


def compute_distance_covered_km(match_id):
    """ total distance per person for one match, in km"""
    path = find_file("positions_raw", match_id)
    total_d_by_person = {}

    context = etree.iterparse(path, events=("end",), tag="FrameSet")
    for _, fs in context:
        team_id = fs.get("TeamId")
        person_id = fs.get("PersonId") or ("BALL" if team_id == "BALL" else None)
        total_d = sum(float(f.get("D")) for f in fs if tag(f) == "Frame" and f.get("D") is not None)
        if person_id:
            total_d_by_person[person_id] = total_d_by_person.get(person_id, 0.0) + total_d
        fs.clear()
        while fs.getprevious() is not None:
            del fs.getparent()[0]

    return {pid: cm / 100 / 1000 for pid, cm in total_d_by_person.items()}


def parse_positions(match_id, out_path, sample_path=None):
    """Streams FrameSets, writes a downsampled Parquet incrementally, and returns summary stats. If sample_path is given, also writes the first 2000 rows of one"""
    path = find_file("positions_raw", match_id)

    writer = None
    schema = pa.schema([
        pa.field("match_id", pa.string()),
        pa.field("game_section", pa.string()),
        pa.field("frame", pa.int64()),
        pa.field("timestamp", pa.string()),  
        pa.field("person_id", pa.string()),
        pa.field("team_id", pa.string()),
        pa.field("x", pa.float64()),
        pa.field("y", pa.float64()),
        pa.field("z", pa.float64()),
        pa.field("d", pa.float64()),
        pa.field("s", pa.float64()),
        pa.field("a", pa.float64()),
        pa.field("minute", pa.int64()),
        pa.field("ball_possession", pa.float64()),
        pa.field("ball_status", pa.float64()),
    ])

    total_d_by_person = {}
    person_team = {}
    ball_x_min, ball_x_max = float("inf"), float("-inf")
    ball_y_min, ball_y_max = float("inf"), float("-inf")
    n_framesets = 0
    sample_rows = []

    writer = pq.ParquetWriter(out_path, schema)
    context = etree.iterparse(path, events=("end",), tag="FrameSet")
    for _, fs in context:
        n_framesets += 1
        df, person_id, total_d_cm = _frameset_to_df(match_id, fs, KEEP_EVERY)

        if person_id:
            total_d_by_person[person_id] = total_d_by_person.get(person_id, 0.0) + total_d_cm
            person_team[person_id] = fs.get("TeamId")

        if fs.get("TeamId") == "BALL" and len(df):
            ball_x_min = min(ball_x_min, df["x"].min())
            ball_x_max = max(ball_x_max, df["x"].max())
            ball_y_min = min(ball_y_min, df["y"].min())
            ball_y_max = max(ball_y_max, df["y"].max())

        if sample_path is not None and len(sample_rows) < 2000:
            sample_rows.extend(df.to_dict("records")[: 2000 - len(sample_rows)])

        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        writer.write_table(table)

        fs.clear()
        while fs.getprevious() is not None:
            del fs.getparent()[0]

    writer.close()

    if sample_path is not None:
        pd.DataFrame(sample_rows, columns=POSITION_COLUMNS).to_csv(sample_path, index=False)

    return {
        "n_framesets": n_framesets,
        "total_d_cm_by_person": total_d_by_person,
        "person_team": person_team,
        "ball_x_range": (ball_x_min, ball_x_max),
        "ball_y_range": (ball_y_min, ball_y_max),
    }


# --- main --------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "positions"), exist_ok=True)

    all_matches = []
    all_players = []
    all_events = []
    total_delete = 0
    total_branched = 0
    position_summaries = {}
    raw_event_counts = {}  # kept + Delete, per match -- for the paper comparison

    for i, match_id in enumerate(MATCH_IDS):
        print(f"\n=== {match_id} ({i+1}/{len(MATCH_IDS)}) ===")

        match_row, player_rows = parse_match_information(match_id)
        all_matches.append(match_row)
        all_players.extend(player_rows)
        print(f"  players: {len(player_rows)}")

        events_df, n_delete, n_branched = parse_events(match_id)
        total_delete += n_delete
        total_branched += n_branched
        raw_event_counts[match_id] = len(events_df) + n_delete
        print(f"  events: {len(events_df)} kept, {n_delete} Delete rows dropped"
              + (f", {n_branched} BRANCHED chains " if n_branched else ""))
        all_events.append(events_df)

        out_parquet = os.path.join(OUT_DIR, "positions", f"{match_id}.parquet")
        sample_csv = os.path.join(OUT_DIR, "positions_sample.csv") if i == 0 else None
        summary = parse_positions(match_id, out_parquet, sample_path=sample_csv)
        position_summaries[match_id] = summary
        print(f"  positions: {summary['n_framesets']} FrameSets streamed -> {out_parquet}")

    matches_df = pd.DataFrame(all_matches)
    players_df = pd.DataFrame(all_players)
    events_df = pd.concat(all_events, ignore_index=True)

    matches_df.to_csv(os.path.join(OUT_DIR, "matches.csv"), index=False)
    players_df.to_csv(os.path.join(OUT_DIR, "players.csv"), index=False)
    events_df.to_csv(os.path.join(OUT_DIR, "events.csv"), index=False)

    print(f"\nTotal Delete events dropped across all matches: {total_delete}")
    print(f"\nTotal BRANCHED events across all matches: {total_branched}")
    print(f"\nRaw event counts (kept + Delete) per match: {raw_event_counts}")

    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
