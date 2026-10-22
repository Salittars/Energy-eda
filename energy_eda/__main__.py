"""Command line: python -m energy_eda report | make-sample"""

import argparse

from . import data, report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="energy_eda", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("report", help="build all charts into figures/")
    rep.add_argument("--source", choices=["sample", "duckdb", "opsd"], default="sample")
    rep.add_argument("--duckdb", dest="duckdb_path", help="path to energy-data-pipeline's data/energy.duckdb")
    rep.add_argument("--opsd-csv", help="local OPSD 60-min CSV (adds solar to --source duckdb)")
    rep.add_argument("--out", default="figures")

    smp = sub.add_parser("make-sample", help="rebuild data/sample/ from the pipeline DuckDB + OPSD CSV")
    smp.add_argument("--duckdb", dest="duckdb_path", required=True)
    smp.add_argument("--opsd-csv", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "report":
        report.run(args.source, args.duckdb_path, args.opsd_csv, args.out)
    elif args.cmd == "make-sample":
        df = data.load_duckdb(args.duckdb_path, args.opsd_csv)
        print(f"wrote {data.write_sample(df)} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
