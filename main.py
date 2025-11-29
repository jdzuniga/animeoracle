import argparse
from src import scrape, clean, train, predict, posters, config, utils


def main(args):
    step = args.step
    grid_search = args.gridsearch
    if step == "scrape":
        scrape.run()
        return
    elif step == "all":
        scrape.run()
        config.RUN_DATE = utils.get_latest_dated_folder(config.DATA_DIR)
        clean.run()
        train.run(grid_search=grid_search)
        predict.run()
        posters.run()
        return

    config.RUN_DATE = utils.get_latest_dated_folder(config.DATA_DIR)
    if step == "train":
        train.run(grid_search=grid_search)
    elif step in ["clean", "predict", "posters"]:
        eval(f'{step}.run()')
    else:
        raise ValueError(f"Unknown step: {step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Jikan ML pipeline")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        help="Pipeline step: scrape | clean | preprocess | train | gridsearch | predict | posters | all"
    )
    parser.add_argument(
        "--gridsearch",
        action="store_true",
        help="Enable grid search mode"
    )
    args = parser.parse_args()
    main(args)

