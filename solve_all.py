import itertools
import json
import math

import numpy as np
import polars as pl


CSV = "Copy of train_V2.csv"
STANDARD = ["duo", "duo-fpp", "solo", "solo-fpp", "squad", "squad-fpp"]


def native(value):
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def rows(frame):
    return frame.to_dicts()


def one(frame):
    return frame.to_dicts()[0]


print("Loading CSV...")
df = pl.read_csv(CSV)
n = df.height
labelled = df.filter(pl.col("winPlacePerc").is_not_null())
std = df.filter(pl.col("matchType").is_in(STANDARD))
labelled_std = labelled.filter(pl.col("matchType").is_in(STANDARD))
answers = {}

# Reusable team and match summaries.
teams = df.group_by(["matchId", "groupId"]).agg(
    pl.len().alias("team_size"),
    pl.col("kills").sum().alias("team_kills"),
    (pl.col("kills").cast(pl.Float64) ** 2).sum().alias("sum_sq_kills"),
    pl.col("assists").sum().alias("team_assists"),
    pl.col("revives").sum().alias("team_revives"),
    pl.col("damageDealt").sum().alias("team_damage"),
    pl.col("winPlacePerc").drop_nulls().n_unique().alias("distinct_outcomes"),
    pl.col("winPlacePerc").drop_nulls().first().alias("team_outcome"),
    pl.col("matchType").first().alias("matchType"),
)
team_count = teams.height

match_ids = sorted(df.get_column("matchId").unique().to_list())
validation_match_ids = set(match_ids[4::5])

print("E1-E25...")
# E1
answers["E1"] = {
    "player_rows": n,
    "matches": len(match_ids),
    "teams": team_count,
}

# E2
missing_outcomes = df.get_column("winPlacePerc").null_count()
answers["E2"] = {
    "missing_outcomes": missing_outcomes,
    "labelled_rows": n - missing_outcomes,
}

# E3
original_rank = pl.col("rankPoints")
answers["E3"] = one(
    df.select(
        pl.len().alias("rows"),
        (original_rank.is_null() | (original_rank == -1)).sum().alias("missing_rankpoints"),
        (
            pl.col("killPoints").is_null()
            | ((pl.col("killPoints") == 0) & (original_rank != -1))
        ).sum().alias("missing_killpoints"),
        (
            pl.col("winPoints").is_null()
            | ((pl.col("winPoints") == 0) & (original_rank != -1))
        ).sum().alias("missing_winpoints"),
    )
)

# E4
e4 = (
    df.filter(
        pl.all_horizontal(
            pl.col("walkDistance").is_not_null(),
            pl.col("rideDistance").is_not_null(),
            pl.col("swimDistance").is_not_null(),
        )
    )
    .with_columns(
        (pl.col("walkDistance") + pl.col("rideDistance") + pl.col("swimDistance")).alias(
            "total_distance"
        )
    )
    .sort(["total_distance", "Id"], descending=[True, False])
    .select(pl.col("Id").alias("id"), "total_distance")
    .head(1)
)
answers["E4"] = one(e4)

# E5
top_players = labelled.filter(pl.col("winPlacePerc") >= 0.8).height
answers["E5"] = {
    "labelled_players": labelled.height,
    "top_players": top_players,
    "top_rate": top_players / labelled.height,
}

# E6
e6 = (
    std.group_by("matchType")
    .agg(pl.col("walkDistance").median().alias("median_walkdistance"))
    .sort(["median_walkdistance", "matchType"], descending=[True, False])
    .head(1)
    .rename({"matchType": "matchtype"})
)
answers["E6"] = one(e6)

# E7
e7df = df.filter(pl.col("kills") >= 5).with_columns(
    (pl.col("headshotKills") / pl.col("kills")).alias("share")
)
answers["E7"] = {
    "rows": e7df.height,
    "mean_headshot_share": e7df.get_column("share").mean(),
    "all_headshot_players": e7df.filter(pl.col("share") == 1).height,
}

# E8
answers["E8"] = {
    "answer_table": rows(
        teams.group_by("team_size")
        .agg(pl.len().alias("teams"))
        .sort("team_size")
    )
}

# E9
answers["E9"] = {
    "players": df.filter(
        (pl.col("kills") == 0) & ((pl.col("assists") > 0) | (pl.col("revives") > 0))
    ).height
}

# E10
answers["E10"] = {
    "answer_table": rows(
        df.group_by("Id")
        .agg(pl.len().alias("appearances"))
        .filter(pl.col("appearances") > 1)
        .sort("Id")
    )
}

# E11
answers["E11"] = {
    "answer_table": rows(
        df.group_by("matchType")
        .agg(pl.len().alias("players"))
        .with_columns((pl.col("players") / n).alias("fraction"))
        .sort(["players", "matchType"], descending=[True, False])
    )
}

# E12
answers["E12"] = one(
    df.with_columns((pl.col("heals") + pl.col("boosts")).alias("consumables"))
    .sort(["consumables", "Id"], descending=[True, False])
    .select(pl.col("Id").alias("id"), "heals", "boosts", "consumables")
    .head(1)
)

# E13
answers["E13"] = one(
    df.filter(pl.col("kills") > 0)
    .sort(["longestKill", "Id"], descending=[True, False])
    .select(pl.col("Id").alias("id"), "kills", pl.col("longestKill").alias("longestkill"))
    .head(1)
)

# E14
c = df.filter(pl.col("weaponsAcquired") == 0).height
answers["E14"] = {"players": c, "fraction": c / n}

# E15
t = df.filter(pl.col("swimDistance") > 0)
answers["E15"] = {
    "players": t.height,
    "mean_swim_distance": t.get_column("swimDistance").mean(),
}

# E16
t = df.filter((pl.col("DBNOs") > 0) & (pl.col("kills") == 0))
answers["E16"] = {"players": t.height, "total_knockdowns": t.get_column("DBNOs").sum()}

# E17: source values have four decimal places; integer scaling gives an exact total.
t = df.filter((pl.col("kills") == 0) & (pl.col("damageDealt") > 0))
damage_units = np.rint(t.get_column("damageDealt").to_numpy() * 10000).astype(np.int64)
total_damage = int(damage_units.sum()) / 10000
answers["E17"] = {
    "players": t.height,
    "total_damage": total_damage,
    "mean_damage": total_damage / t.height,
}

# E18
k = df.get_column("kills")
answers["E18"] = {
    "answer_table": [
        {"kill_band": "0", "players": int((k == 0).sum())},
        {"kill_band": "1-2", "players": int(((k >= 1) & (k <= 2)).sum())},
        {"kill_band": "3-5", "players": int(((k >= 3) & (k <= 5)).sum())},
        {"kill_band": "6+", "players": int((k >= 6).sum())},
    ]
}

# E19
answers["E19"] = {
    "answer_table": rows(
        std.group_by("matchType")
        .agg(pl.len().alias("players"), pl.col("heals").sum().alias("total_heals"))
        .sort("matchType")
    )
}

# E20
t = df.filter(pl.col("teamKills") > 0)
answers["E20"] = {"players": t.height, "total_team_kills": t.get_column("teamKills").sum()}

# E21
t = df.filter(pl.col("roadKills") > 0)
answers["E21"] = {
    "players": t.height,
    "total_road_kills": t.get_column("roadKills").sum(),
    "maximum_road_kills": t.get_column("roadKills").max(),
}

# E22
c = df.filter(pl.col("weaponsAcquired") >= 10).height
answers["E22"] = {"players": c, "fraction": c / n}

# E23
q = df.select(
    pl.col("walkDistance").quantile(0.25, interpolation="linear").alias("q1"),
    pl.col("walkDistance").quantile(0.50, interpolation="linear").alias("median"),
    pl.col("walkDistance").quantile(0.75, interpolation="linear").alias("q3"),
).row(0, named=True)
q["iqr"] = q["q3"] - q["q1"]
answers["E23"] = q

# E24
c = df.filter((pl.col("heals") > 0) & (pl.col("boosts") > 0)).height
answers["E24"] = {"players": c, "fraction": c / n}

# E25
t = df.filter((pl.col("rideDistance") > 0) & (pl.col("walkDistance") == 0))
answers["E25"] = {
    "players": t.height,
    "median_ride_distance": t.get_column("rideDistance").median(),
}

print("M1-M25...")
# M1
eligible_teams = teams.filter(pl.col("distinct_outcomes") == 1)
eligible_keys = eligible_teams.select("matchId", "groupId")
eligible_players = labelled.join(eligible_keys, on=["matchId", "groupId"], how="inner")
answers["M1"] = {
    "player_mean": eligible_players.get_column("winPlacePerc").mean(),
    "team_mean": eligible_teams.get_column("team_outcome").mean(),
    "eligible_teams": eligible_teams.height,
    "excluded_teams": team_count - eligible_teams.height,
}

# M2
defined = teams.filter(pl.col("team_kills") > 0).with_columns(
    (pl.col("sum_sq_kills") / (pl.col("team_kills") ** 2)).alias("concentration")
)
answers["M2"] = {
    "rows": team_count,
    "undefined_teams": team_count - defined.height,
    "mean_concentration": defined.get_column("concentration").mean(),
}

# M3
m3 = df.select("matchId", "damageDealt").with_columns(
    pl.col("damageDealt").rank(method="average").over("matchId").alias("rank"),
    pl.len().over("matchId").alias("match_n"),
).with_columns(
    pl.when(pl.col("match_n") == 1)
    .then(0.5)
    .otherwise((pl.col("rank") - 1) / (pl.col("match_n") - 1))
    .alias("pct")
)
answers["M3"] = {
    "rows": m3.height,
    "mean_damage_percentile": m3.get_column("pct").mean(),
    "players_at_or_above_90th": m3.filter(pl.col("pct") >= 0.9).height,
}
del m3

# M4
answers["M4"] = {
    "answer_table": rows(
        teams.filter(pl.col("distinct_outcomes") > 1)
        .select("matchId", "groupId", "distinct_outcomes")
        .sort(["matchId", "groupId"])
    )
}

# M5: exact 0.0001 metre integer units.
squad_team_walk = (
    df.filter(pl.col("matchType") == "squad-fpp")
    .with_columns((pl.col("walkDistance") * 10000).round(0).cast(pl.Int64).alias("walk_units"))
    .group_by(["matchId", "groupId"])
    .agg(
        pl.col("walk_units").sum().alias("walk_units"),
        pl.col("winPlacePerc").drop_nulls().n_unique().alias("distinct_outcomes"),
        pl.col("winPlacePerc").drop_nulls().first().alias("team_outcome"),
    )
    .filter(pl.col("distinct_outcomes") == 1)
)
median_units = squad_team_walk.get_column("walk_units").median()
m5_table = []
for name, condition in [
    ("below_or_equal", pl.col("walk_units") <= median_units),
    ("above", pl.col("walk_units") > median_units),
]:
    g = squad_team_walk.filter(condition)
    top = g.filter(pl.col("team_outcome") >= 0.8).height
    m5_table.append(
        {
            "walk_group": name,
            "median_team_walk": median_units / 10000,
            "teams": g.height,
            "top_teams": top,
            "top_rate": top / g.height,
        }
    )
answers["M5"] = {"answer_table": m5_table}

# M6
winners = df.filter(pl.col("winPlacePerc") == 1)
answers["M6"] = {
    "winning_players": winners.height,
    "winning_teams": winners.select(pl.struct(["matchId", "groupId"]).n_unique()).item(),
    "malformed_matches_excluded": "no",
}

# M7
is_val = df.get_column("matchId").is_in(list(validation_match_ids))
answers["M7"] = {
    "rows": len(match_ids),
    "train_matches": len(match_ids) - len(validation_match_ids),
    "validation_matches": len(validation_match_ids),
    "train_players": int((~is_val).sum()),
    "validation_players": int(is_val.sum()),
}

# M8
m8 = df.group_by("matchId").agg(
    pl.col("groupId").n_unique().alias("observed_groups"),
    pl.col("numGroups").n_unique().alias("reported_distinct"),
    pl.col("numGroups").first().alias("reported_groups"),
)
inconsistent = m8.filter(pl.col("reported_distinct") != 1).height
comparable = m8.filter(pl.col("reported_distinct") == 1)
matching = comparable.filter(pl.col("observed_groups") == pl.col("reported_groups")).height
answers["M8"] = {
    "matches": m8.height,
    "inconsistent_matches": inconsistent,
    "comparable_matches": comparable.height,
    "matching_matches": matching,
    "mismatching_matches": comparable.height - matching,
}

# M9
squad = df.filter(pl.col("matchType") == "squad-fpp")
m9teams = squad.group_by(["matchId", "groupId"]).agg(
    pl.len().alias("team_size"), pl.col("damageDealt").sum().alias("team_damage")
).filter((pl.col("team_size") >= 2) & (pl.col("team_damage") > 0))
answers["M9"] = {
    "answer_table": rows(
        squad.join(m9teams, on=["matchId", "groupId"], how="inner")
        .with_columns((pl.col("damageDealt") / pl.col("team_damage")).alias("damage_share"))
        .sort(
            ["damage_share", "matchId", "groupId", "Id"],
            descending=[True, False, False, False],
        )
        .select("matchId", "groupId", "Id", "damageDealt", "team_damage", "damage_share")
        .head(5)
    )
}

# M10
m10_table = []
for name, lo, hi, include_hi in [
    ("low", 0.0, 0.2, False),
    ("middle", 0.2, 0.8, False),
    ("top", 0.8, 1.0, True),
]:
    cond = (pl.col("winPlacePerc") >= lo) & (
        (pl.col("winPlacePerc") <= hi) if include_hi else (pl.col("winPlacePerc") < hi)
    )
    g = labelled.filter(cond).with_columns((pl.col("heals") + pl.col("boosts")).alias("cons"))
    m10_table.append(
        {"placement_band": name, "players": g.height, "mean_consumables": g.get_column("cons").mean()}
    )
answers["M10"] = {"answer_table": m10_table}

# M11
m11 = std.filter(pl.col("kills") >= 5).with_columns(
    (pl.col("headshotKills") / pl.col("kills")).alias("individual_share")
).group_by("matchType").agg(
    pl.len().alias("players"),
    (pl.col("headshotKills").sum() / pl.col("kills").sum()).alias("pooled_headshot_share"),
    pl.col("individual_share").mean().alias("mean_player_headshot_share"),
).sort("matchType")
answers["M11"] = {"answer_table": rows(m11)}

# M12
leaders = (
    std.sort(["matchId", "kills", "damageDealt", "Id"], descending=[False, True, True, False])
    .unique(subset=["matchId"], keep="first", maintain_order=True)
)
m12 = leaders.group_by("matchType").agg(
    pl.len().alias("leaders"),
    pl.col("winPlacePerc").count().alias("labelled_leaders"),
    pl.col("winPlacePerc").mean().alias("mean_placement"),
    (pl.col("winPlacePerc") >= 0.8).sum().alias("top_count"),
).with_columns((pl.col("top_count") / pl.col("labelled_leaders")).alias("top_rate"))
answers["M12"] = {"answer_table": rows(m12.select(pl.exclude("top_count")).sort("matchType"))}

# M13
m13_table = []
for mode in STANDARD:
    g = labelled_std.filter(pl.col("matchType") == mode).select("walkDistance", "winPlacePerc")
    ranked = g.with_columns(
        pl.col("walkDistance").rank(method="average").alias("rw"),
        pl.col("winPlacePerc").rank(method="average").alias("ry"),
    )
    corr = ranked.select(pl.corr("rw", "ry")).item() if g.height >= 2 else None
    m13_table.append({"matchType": mode, "players": g.height, "spearman": corr})
answers["M13"] = {"answer_table": m13_table}

# M14
m14teams = (
    squad.group_by(["matchId", "groupId"])
    .agg(
        pl.len().alias("team_size"),
        pl.col("revives").sum().alias("team_revives"),
        pl.col("winPlacePerc").drop_nulls().n_unique().alias("distinct_outcomes"),
        pl.col("winPlacePerc").drop_nulls().first().alias("team_outcome"),
    )
    .filter((pl.col("team_size") == 4) & (pl.col("distinct_outcomes") == 1))
)
m14_table = []
for name, cond in [
    ("zero", pl.col("team_revives") == 0),
    ("positive", pl.col("team_revives") > 0),
]:
    g = m14teams.filter(cond)
    top = g.filter(pl.col("team_outcome") >= 0.8).height
    m14_table.append({"revive_group": name, "teams": g.height, "top_teams": top, "top_rate": top / g.height})
answers["M14"] = {"answer_table": m14_table}

# M15
mode_means = labelled_std.group_by("matchType").agg(pl.col("winPlacePerc").mean().alias("mean"))
pooled = labelled_std.get_column("winPlacePerc").mean()
equal = mode_means.get_column("mean").mean() if mode_means.height == 6 else None
answers["M15"] = {
    "pooled_player_mean": pooled,
    "equal_mode_mean": equal,
    "difference": None if equal is None else equal - pooled,
}

# M16
m16teams = teams.filter(pl.col("team_size") >= 2).select(
    "matchId", "groupId", "team_size", "team_kills"
)
m16 = df.join(m16teams, on=["matchId", "groupId"], how="inner").with_columns(
    ((pl.col("team_kills") - pl.col("kills")) / (pl.col("team_size") - 1)).alias(
        "mean_teammate_kills"
    )
).with_columns((pl.col("kills") - pl.col("mean_teammate_kills")).alias("kill_gap"))
answers["M16"] = {
    "answer_table": rows(
        m16.sort(["kill_gap", "matchId", "groupId", "Id"], descending=[True, False, False, False])
        .select("matchId", "groupId", "Id", "kills", "mean_teammate_kills", "kill_gap")
        .head(5)
    )
}

# M17
first_match = match_ids[0]
m17 = df.filter(pl.col("matchId") == first_match)
match_mean = m17.get_column("damageDealt").mean()
match_std = m17.get_column("damageDealt").std(ddof=0)
m17 = m17.with_columns(
    pl.lit(match_mean).alias("match_mean"), pl.lit(match_std).alias("match_std")
).with_columns(
    pl.when(pl.lit(match_std) == 0)
    .then(0.0)
    .otherwise((pl.col("damageDealt") - match_mean) / match_std)
    .alias("damage_zscore")
)
answers["M17"] = {
    "answer_table": rows(
        m17.sort("Id")
        .select("matchId", "Id", "damageDealt", "match_mean", "match_std", "damage_zscore")
        .head(5)
    )
}

# M18
m18 = labelled.filter(pl.col("maxPlace") > 1).with_columns(
    (
        pl.col("winPlacePerc")
        - ((pl.col("winPlacePerc") * (pl.col("maxPlace") - 1)).round(0) / (pl.col("maxPlace") - 1))
    ).abs().alias("grid_distance")
)
answers["M18"] = {
    "eligible_players": m18.height,
    "outside_tolerance": m18.filter(pl.col("grid_distance") > 0.000050001).height,
    "maximum_grid_distance": m18.get_column("grid_distance").max(),
}

# M19
m19matches = df.group_by("matchId").agg(
    pl.col("matchType").n_unique().alias("type_n"),
    pl.col("matchType").first().alias("matchType"),
    pl.col("matchDuration").n_unique().alias("duration_n"),
    pl.col("matchDuration").first().alias("matchDuration"),
).filter(
    (pl.col("type_n") == 1)
    & (pl.col("duration_n") == 1)
    & pl.col("matchType").is_in(STANDARD)
)
answers["M19"] = {
    "answer_table": rows(
        m19matches.group_by("matchType")
        .agg(
            pl.len().alias("matches"),
            pl.col("matchDuration").mean().alias("mean_duration"),
            pl.col("matchDuration").median().alias("median_duration"),
        )
        .sort("matchType")
    )
}

# M20
kill_leaders = (
    df.sort(["matchId", "kills", "Id"], descending=[False, True, False])
    .unique("matchId", keep="first", maintain_order=True)
    .select("matchId", pl.col("Id").alias("kill_id"))
)
damage_leaders = (
    df.sort(["matchId", "damageDealt", "Id"], descending=[False, True, False])
    .unique("matchId", keep="first", maintain_order=True)
    .select("matchId", pl.col("Id").alias("damage_id"))
)
m20 = kill_leaders.join(damage_leaders, on="matchId")
same = m20.filter(pl.col("kill_id") == pl.col("damage_id")).height
answers["M20"] = {
    "matches": m20.height,
    "same_leader": same,
    "different_leader": m20.height - same,
    "same_fraction": same / m20.height,
}

# M21
quant = std.group_by("matchType").agg(
    pl.col("walkDistance").quantile(0.25, interpolation="linear").alias("q1"),
    pl.col("walkDistance").quantile(0.75, interpolation="linear").alias("q3"),
    pl.len().alias("players"),
).with_columns((pl.col("q3") + 1.5 * (pl.col("q3") - pl.col("q1"))).alias("upper_fence"))
m21 = std.join(quant.select("matchType", "upper_fence"), on="matchType").group_by(
    "matchType"
).agg((pl.col("walkDistance") > pl.col("upper_fence")).sum().alias("above_fence"))
m21 = quant.join(m21, on="matchType").with_columns(
    (pl.col("above_fence") / pl.col("players")).alias("above_fraction")
).select("matchType", "q1", "q3", "upper_fence", "players", "above_fence", "above_fraction").sort("matchType")
answers["M21"] = {"answer_table": rows(m21)}

# M22
def boost_stats(frame, name):
    z = frame.filter(pl.col("boosts") == 0)
    b = frame.filter(pl.col("boosts") >= 1)
    zm = z.get_column("winPlacePerc").mean()
    bm = b.get_column("winPlacePerc").mean()
    return {
        "matchType": name,
        "zero_players": z.height,
        "boosted_players": b.height,
        "zero_mean": zm,
        "boosted_mean": bm,
        "difference": bm - zm,
    }

m22_table = [boost_stats(labelled_std.filter(pl.col("matchType") == mode), mode) for mode in STANDARD]
pooled_boost = boost_stats(labelled_std, "ALL_PLAYERS")
equal_diff = float(np.mean([x["difference"] for x in m22_table]))
m22_table.extend(
    [
        pooled_boost,
        {
            "matchType": "EQUAL_MODES",
            "zero_players": None,
            "boosted_players": None,
            "zero_mean": None,
            "boosted_mean": None,
            "difference": equal_diff,
        },
    ]
)
answers["M22"] = {"answer_table": m22_table}

# M23
m23 = squad.group_by(["matchId", "groupId"]).agg(
    pl.len().alias("team_size"),
    pl.col("kills").sum().alias("team_kills"),
    pl.col("assists").sum().alias("team_assists"),
).filter(pl.col("team_size") == 4)
m23pos = m23.filter(pl.col("team_kills") > 0).with_columns(
    (pl.col("team_assists") / pl.col("team_kills")).alias("ratio")
)
answers["M23"] = {
    "teams": m23.height,
    "zero_kill_teams": m23.filter(pl.col("team_kills") == 0).height,
    "mean_assists_per_kill": m23pos.get_column("ratio").mean(),
    "pooled_assists_per_kill": m23pos.get_column("team_assists").sum()
    / m23pos.get_column("team_kills").sum(),
}

# M24
m24 = labelled.filter(pl.col("matchType") == "solo-fpp").select("damageDealt", "winPlacePerc")
m24 = m24.with_columns(
    (pl.col("winPlacePerc") >= 0.8).alias("positive"),
    pl.col("damageDealt").rank(method="average").alias("damage_rank"),
)
npos = m24.filter(pl.col("positive")).height
nneg = m24.height - npos
rank_sum_pos = m24.filter(pl.col("positive")).get_column("damage_rank").sum()
auc = (rank_sum_pos - npos * (npos + 1) / 2) / (npos * nneg)
answers["M24"] = {"positive_players": npos, "negative_players": nneg, "damage_auc": auc}

# M25
duo = df.filter(pl.col("matchType") == "duo-fpp")
m25 = duo.group_by(["matchId", "groupId"]).agg(
    pl.len().alias("team_size"),
    pl.col("kills").min().alias("min_kills"),
    pl.col("kills").max().alias("max_kills"),
    pl.col("winPlacePerc").drop_nulls().n_unique().alias("distinct_outcomes"),
    pl.col("winPlacePerc").drop_nulls().first().alias("team_outcome"),
).filter((pl.col("team_size") == 2) & (pl.col("distinct_outcomes") == 1)).with_columns(
    (pl.col("max_kills") - pl.col("min_kills")).alias("kill_gap")
)
m25_table = []
for name, cond in [("equal", pl.col("kill_gap") == 0), ("unequal", pl.col("kill_gap") > 0)]:
    g = m25.filter(cond)
    top = g.filter(pl.col("team_outcome") >= 0.8).height
    m25_table.append(
        {
            "kill_balance": name,
            "teams": g.height,
            "mean_kill_gap": g.get_column("kill_gap").mean(),
            "mean_placement": g.get_column("team_outcome").mean(),
            "top_rate": top / g.height,
        }
    )
answers["M25"] = {"answer_table": m25_table}

print("H1-H6...")
# H1
h1p = df.filter(pl.col("matchId") == first_match)
h1 = h1p.group_by("groupId").agg(pl.col("walkDistance").sum().alias("team_walk")).with_columns(
    (pl.col("team_walk").rank(method="average") / pl.len()).alias("predicted_score")
).sort(["predicted_score", "groupId"]).head(4)
legal = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
pred = h1.get_column("predicted_score").to_list()
candidates = []
for seq in itertools.combinations(legal, 4):
    cost = sum((a - b) ** 2 for a, b in zip(pred, seq))
    candidates.append((cost, seq))
min_cost, assigned = min(candidates, key=lambda x: (round(x[0], 15), x[1]))
h1_table = []
for row, value in zip(h1.select("groupId", "predicted_score").to_dicts(), assigned):
    h1_table.append(
        {
            "matchId": first_match,
            "groupId": row["groupId"],
            "predicted_score": row["predicted_score"],
            "assigned_value": value,
            "minimum_squared_error": min_cost,
        }
    )
answers["H1"] = {"answer_table": h1_table}

# H2
h2_ids = set(match_ids[0::50])
h2 = labelled.filter(pl.col("matchId").is_in(list(h2_ids))).with_columns(
    (pl.col("walkDistance").rank(method="average").over("matchId") / pl.len().over("matchId")).alias(
        "pred_b"
    )
).with_columns(
    (pl.col("winPlacePerc") - 0.5).abs().alias("err_a"),
    (pl.col("winPlacePerc") - pl.col("pred_b")).abs().alias("err_b"),
)
h2matches = h2.group_by("matchId").agg(
    pl.col("err_a").mean().alias("mae_a"), pl.col("err_b").mean().alias("mae_b")
)
match_a = h2matches.get_column("mae_a").mean()
match_b = h2matches.get_column("mae_b").mean()
player_a = h2.get_column("err_a").mean()
player_b = h2.get_column("err_b").mean()
match_winner = "A" if match_a < match_b else "B" if match_b < match_a else "tie"
player_winner = "A" if player_a < player_b else "B" if player_b < player_a else "tie"
answers["H2"] = {
    "matches": h2matches.height,
    "players": h2.height,
    "match_mae_a": match_a,
    "match_mae_b": match_b,
    "player_mae_a": player_a,
    "player_mae_b": player_b,
    "match_winner": match_winner,
    "player_winner": player_winner,
    "ranking_changes": "yes" if match_winner != player_winner else "no",
}

# H3
four = teams.filter(pl.col("team_size") == 4).sort(["matchId", "groupId"])
h3match = four.get_column("matchId")[0]
h3group = four.filter(pl.col("matchId") == h3match).get_column("groupId").min()
h3 = df.filter((pl.col("matchId") == h3match) & (pl.col("groupId") == h3group)).with_columns(
    (2 * pl.col("kills") + pl.col("assists") + 3 * pl.col("revives")).alias("score")
)
base_total = h3.get_column("score").sum()
h3 = h3.with_columns(
    pl.when(pl.lit(base_total) == 0)
    .then(25.0)
    .otherwise(100 * pl.col("score") / base_total)
    .alias("allocation")
).sort(["allocation", "Id"])
selected = h3.row(0, named=True)
allocations = []
for extra in range(4):
    selected_score = selected["score"] + 3 * extra
    total_score = base_total + 3 * extra
    allocation = 25.0 if total_score == 0 else 100 * selected_score / total_score
    allocations.append(allocation)
answers["H3"] = {
    "matchid": h3match,
    "groupid": h3group,
    "id": selected["Id"],
    "extra_0_allocation": allocations[0],
    "extra_1_allocation": allocations[1],
    "extra_2_allocation": allocations[2],
    "extra_3_allocation": allocations[3],
    "maximum_allocation": max(allocations),
}

# H4
h4 = labelled.filter(pl.col("matchType") == "squad-fpp")
h4matches = h4.group_by("matchId").agg(
    (pl.col("revives") == 0).sum().alias("zero_n"),
    (pl.col("revives") > 0).sum().alias("pos_n"),
    pl.col("winPlacePerc").filter(pl.col("revives") == 0).mean().alias("zero_mean"),
    pl.col("winPlacePerc").filter(pl.col("revives") > 0).mean().alias("pos_mean"),
).filter((pl.col("zero_n") > 0) & (pl.col("pos_n") > 0)).sort("matchId").head(12).with_columns(
    (pl.col("pos_mean") - pl.col("zero_mean")).alias("difference")
)
diffs = np.array(h4matches.get_column("difference").to_list(), dtype=float)
observed = float(diffs.mean())
permuted = []
for signs in itertools.product([-1.0, 1.0], repeat=len(diffs)):
    permuted.append(float(np.mean(diffs * np.asarray(signs))))
extreme = sum(abs(x) >= abs(observed) - 1e-12 for x in permuted)
answers["H4"] = {
    "matches": len(diffs),
    "observed_mean_difference": observed,
    "extreme_sign_patterns": extreme,
    "total_sign_patterns": len(permuted),
    "two_sided_p_value": extreme / len(permuted),
}

# H5
h5 = df.select("matchId", "walkDistance", "boosts", "kills", "winPlacePerc").with_columns(
    pl.col("matchId").is_in(list(validation_match_ids)).alias("is_val")
).filter(pl.col("winPlacePerc").is_not_null())
train = h5.filter(~pl.col("is_val"))
valid = h5.filter(pl.col("is_val"))
feature_cols = ["walkDistance", "boosts", "kills"]
Xtr_raw = np.column_stack([np.log1p(train.get_column(c).to_numpy()) for c in feature_cols])
Xva_raw = np.column_stack([np.log1p(valid.get_column(c).to_numpy()) for c in feature_cols])
ytr = train.get_column("winPlacePerc").to_numpy()
yva = valid.get_column("winPlacePerc").to_numpy()
means = Xtr_raw.mean(axis=0)
stds = Xtr_raw.std(axis=0, ddof=0)
stds[stds == 0] = 1
Ztr = (Xtr_raw - means) / stds
Zva = (Xva_raw - means) / stds
intercept = float(ytr.mean())
beta = np.linalg.solve(Ztr.T @ Ztr + 10 * np.eye(3), Ztr.T @ (ytr - intercept))
ridge_pred = np.clip(intercept + Zva @ beta, 0, 1)
baseline_pred = np.full(len(yva), intercept)
valid_matches_np = valid.get_column("matchId").to_numpy()

def match_weighted_mae(actual, predicted, groups):
    order = np.argsort(groups)
    gs = groups[order]
    err = np.abs(actual[order] - predicted[order])
    _, starts, counts = np.unique(gs, return_index=True, return_counts=True)
    sums = np.add.reduceat(err, starts)
    return float(np.mean(sums / counts))

ridge_mae = match_weighted_mae(yva, ridge_pred, valid_matches_np)
base_mae = match_weighted_mae(yva, baseline_pred, valid_matches_np)
answers["H5"] = {
    "train_players": train.height,
    "validation_players": valid.height,
    "intercept": intercept,
    "beta_walkdistance": beta[0],
    "beta_boosts": beta[1],
    "beta_kills": beta[2],
    "ridge_match_mae": ridge_mae,
    "baseline_match_mae": base_mae,
    "better_predictor": "ridge" if ridge_mae < base_mae else "baseline" if base_mae < ridge_mae else "tie",
}

# H6
h6base = labelled.join(eligible_keys, on=["matchId", "groupId"], how="inner")

def leakage_result(base, split_name, val_condition):
    marked = base.with_columns(val_condition.alias("is_val"))
    tr = marked.filter(~pl.col("is_val"))
    va = marked.filter(pl.col("is_val"))
    trteams = tr.select("matchId", "groupId").unique().with_columns(pl.lit(True).alias("has_teammate"))
    va = va.join(trteams, on=["matchId", "groupId"], how="left").with_columns(
        pl.col("has_teammate").fill_null(False)
    )
    train_mean = tr.get_column("winPlacePerc").mean()
    va = va.with_columns(
        pl.when(pl.col("has_teammate"))
        .then(pl.col("winPlacePerc"))
        .otherwise(train_mean)
        .alias("prediction")
    )
    leaked = va.filter(pl.col("has_teammate")).height
    return {
        "split": split_name,
        "train_players": tr.height,
        "validation_players": va.height,
        "validation_with_training_teammate": leaked,
        "teammate_leakage_rate": leaked / va.height,
        "player_mae": va.select((pl.col("prediction") - pl.col("winPlacePerc")).abs().mean()).item(),
    }

h6sorted = h6base.sort("Id").with_row_index("position", offset=1)
player_res = leakage_result(h6sorted, "player", (pl.col("position") % 5) == 0)
match_res = leakage_result(
    h6base,
    "match",
    pl.col("matchId").is_in(list(validation_match_ids)),
)
answers["H6"] = {"answer_table": [player_res, match_res]}

# Internal consistency checks.
assert sum(r["teams"] for r in answers["E8"]["answer_table"]) == team_count
assert sum(r["players"] for r in answers["E18"]["answer_table"]) == n
assert answers["M7"]["train_players"] + answers["M7"]["validation_players"] == n
assert answers["M8"]["matching_matches"] + answers["M8"]["mismatching_matches"] == answers["M8"]["comparable_matches"]

answers = native(answers)
with open("answers.json", "w", encoding="utf-8") as f:
    json.dump(answers, f, indent=2, ensure_ascii=False, allow_nan=False)

print(json.dumps(answers, indent=2, ensure_ascii=False, allow_nan=False))
