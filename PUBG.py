import json
import polars as pl

file_path = "Copy of train_V2.csv"

# Lazy loading avoids reading the entire dataset into memory immediately.
df = pl.scan_csv(file_path)

# E1: rows, distinct matches, and distinct (matchId, groupId) teams
e1 = (
    df.select(
        pl.len().alias("player_rows"),
        pl.col("matchId").n_unique().alias("matches"),
        pl.struct(["matchId", "groupId"]).n_unique().alias("teams"),
    )
    .collect(engine="streaming")
    .to_dicts()[0]
)

# E2: missing and non-missing winPlacePerc values
e2 = (
    df.select(
        pl.col("winPlacePerc").null_count().alias("missing_outcomes"),
        pl.col("winPlacePerc").is_not_null().sum().alias("labelled_rows"),
    )
    .collect(engine="streaming")
    .to_dicts()[0]
)

# E3: preserve the original rankPoints before repairing columns
repaired = (
    df.with_columns(
        pl.col("rankPoints").alias("original_rankPoints")
    )
    .with_columns(
        # Convert rankPoints == -1 to missing
        pl.when(pl.col("rankPoints") == -1)
        .then(None)
        .otherwise(pl.col("rankPoints"))
        .alias("rankPoints"),

        # Convert zero killPoints to missing only if original rankPoints != -1
        pl.when(
            (pl.col("killPoints") == 0)
            & (pl.col("original_rankPoints") != -1)
        )
        .then(None)
        .otherwise(pl.col("killPoints"))
        .alias("killPoints"),

        # Convert zero winPoints to missing only if original rankPoints != -1
        pl.when(
            (pl.col("winPoints") == 0)
            & (pl.col("original_rankPoints") != -1)
        )
        .then(None)
        .otherwise(pl.col("winPoints"))
        .alias("winPoints"),
    )
)

e3 = (
    repaired.select(
        pl.len().alias("rows"),
        pl.col("rankPoints").null_count().alias("missing_rankpoints"),
        pl.col("killPoints").null_count().alias("missing_killpoints"),
        pl.col("winPoints").null_count().alias("missing_winpoints"),
    )
    .collect(engine="streaming")
    .to_dicts()[0]
)

print("E1:", json.dumps(e1))
print("E2:", json.dumps(e2))
print("E3:", json.dumps(e3))