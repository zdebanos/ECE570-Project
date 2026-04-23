# ECE 57000 Project

The only requirement is to have the `uv` project manager installed. Then run:

```bash
uv sync
uv run solar_scheduling.py < refin.txt
```

The package manager should download all needed dependencies.

The `solar_scheduling.py` file calls both the forecast GRU neural network and the MILP scheduler. The `refin.txt` contains the reference input for energy scheduling in the centre of Budapest at 12 a.m. on February 12th, 2026. You may try to input different locations (latitude and longitude) and a different date.
> Please note that this script uses OpenMeteo recent data and remembers approximately 3 months of past data.

The project is divided into packages, the most important are `solar-gru` and `schedule-model`.
The `solar-gru` contains the already pretrained weights and, class definition for the forecast model, dataset means, standard deviations and the Jupyter Notebook on which the model was trained inside Google Colab.
