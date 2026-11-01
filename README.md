# Appliance Industry: Regression, Neural-Network and ARIMA Modelling

Statistical modelling project for the **10th National College Student Statistical Modeling Competition (Shanghai Region, Second Prize)**. The project analyses the development level of China's home appliance industry using regression and neural-network models, and forecasts future trends with ARIMA.

## Project overview

The project proposes an economic-growth measurement approach for China's home appliance industry and tests it empirically. It combines three modelling components:

1. **Descriptive analysis** — indicator selection, measurement-method choice, and descriptive statistics on the cleaned dataset.
2. **Regression and neural-network fitting** — both models are fitted to the indicator data to examine how domestic and overseas demand drive industry development.
3. **ARIMA forecasting** — a time-series model projects the next five periods to assess future trends.

The main finding is that domestic demand drives sales growth, with domestic demand growth exceeding overseas demand. This pushes the industry toward reasonable inventory levels and improved production-sales coordination, with an overall stable and positive trend. Product innovation and market development are identified as the main directions for future growth.

## My contribution

I was responsible for the **regression analysis and ARIMA forecasting components**:

- Selected 3,000 relevant observations from over 30,000 raw records and performed data cleaning.
- Performed descriptive statistics in SPSS.
- Constructed and compared linear regression specifications, and interpreted the fitted models for sales, production, and inventory.
- Applied the ARIMA model to forecast the next five periods and analyzed the future development trends.

The neural-network fitting was completed as part of the team's joint modelling work. I contributed to interpreting the combined results of all models.

## Figures

| Figure | What it shows |
|---|---|
| `figures/01-spss-time-series-forecast.png` | Time-series forecast for sales volume (SPSS): actual values, fitted values, and future prediction. **This is my core contribution.** |
| `figures/02-spss-linear-regression-fit.png` | Linear regression fit for sales, production, and inventory (SPSS). **This is my core contribution.** |
| `figures/03-matlab-nn-time-series.png` | Neural-network time-series response (MATLAB): training, validation, and test fit. |
| `figures/04-matlab-nn-performance.png` | Neural-network training performance (MSE) across epochs (MATLAB). |
| `figures/05-matlab-nn-regression.png` | Neural-network regression plot (R values) comparing network outputs to targets (MATLAB). |

## Code

- `matlab/feedforward-net-fitting.m` — feedforward neural-network fitting (refrigerator data), MATLAB `fitnet`.
- `matlab/narx-time-series-forecast.m` — NARX nonlinear autoregressive network for time-series forecasting (air-conditioner data), MATLAB `narxnet`.

Both MATLAB scripts are excerpts from the competition paper and belong to the team's neural-network modelling component. My regression and ARIMA components were carried out in SPSS through its graphical interface and have no standalone script; the model summary figures document the fitted specifications.

## Data availability

The dataset contains non-public commercial information. Raw data is therefore not included in this repository. The modelling code and figures are shared to document the methods and results; the data itself cannot be redistributed.

## Methods and Tools

- **Methods**: Multiple Linear Regression, Neural Networks, ARIMA (time-series forecasting)
- **Tools**: SPSS, MATLAB
