import pandas as pd
import warnings
import argparse
from src.transform_predict import transform, score

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="input csv file path, e.g. data/example_input.csv",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output csv file path, e.g. ./example_output.csv",
        required=True,
    )
    parser.add_argument(
        "-sm",
        "--score_month",
        default=None,
        help="Specify the Month to be scored, by default it will be the latest month in the dataset.",
        required=False,
    )
    parser.add_argument(
        "-sy",
        "--score_year",
        default=None,
        help="Specify the year of the month to be scored, by default it will be the latest year in the dataset.",
        required=False,
    )
    parser.add_argument(
        "-l",
        "--lag",
        default=2,
        help="Specify the number of lag months to be calculated for the feature pipeline, by default this will be 2 which is required for CBA_Long model. For CBA_Short model, this needs to be set to 0",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model_path",
        default='models/CBA_AITD_Long.zip',
        help="Specify the number of lag months to be calculated for the feature pipeline, by default this will be 2 which is required for CBA_Long model. For CBA_Short model, this needs to be set to 0",
        required=False,
    )
    args = parser.parse_args()
    
    if args.score_month:
        score_month = int(args.score_month)
    else: 
        score_month = args.score_month
    
    df = pd.read_csv(args.input)
    transformed_df = transform(df, score_month = args.score_month, lag=int(args.lag))
    prediction_df = score(transformed_df, model_loc=args.model_path)
    prediction_df.to_csv(args.output, index=False)
    print(f"----Finish, file save to {args.output}")