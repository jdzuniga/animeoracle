import argparse
from src import config, scrape, clean_data, train, predict, get_posters


def main(step):
    config.create_directories()

    if step == "scrape":
        scrape.run()
    elif step == "clean":
        clean_data.run()
    elif step == "train":
        train.run()
    elif step == "predict":
        predict.run()
    elif step == "posters":
        get_posters.run()
    elif step == "all":
        scrape.run()
        clean_data.run()
        train.run()
        predict.run()
        get_posters.run()
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
    main(args.step)

