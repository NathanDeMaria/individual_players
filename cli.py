from pathlib import Path
from fire import Fire

from individual_players import fit_params


class _Cli:
    @staticmethod
    def fit_params(values_csv_path: str):
        """
        Create the VPP bayesian update params file
        given a "values" .csv file
        """
        params, *_ = fit_params(values_csv_path)
        path = Path(values_csv_path)
        param_file = f"{path.stem.rstrip('_values')}_params.json"
        with open(path.parent / param_file, "w") as file:
            file.write(params.to_json())


if __name__ == "__main__":
    Fire(_Cli)
