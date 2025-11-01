import argparse
from src import scrape, clean, train, predict, posters, config, utils


def main(args):
    step = args.step
    if step == "scrape":
        scrape.run()
    elif step in ["clean", "train", "predict", "posters"]:
        config.RUN_DATE = utils.get_latest_dated_folder(config.DATA_DIR)
        eval(f'{step}.run()')

    elif step == "all":
        scrape.run()
        config.RUN_DATE = utils.get_latest_dated_folder(config.DATA_DIR)
        clean.run()
        train.run()
        predict.run()
        posters.run()
    else:
        raise ValueError(f"Unknown step: {step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Jikan ML pipeline")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        help="Pipeline step: scrape | clean | preprocess | train | predict | posters | all"
    )
    args = parser.parse_args()
    main(args)

