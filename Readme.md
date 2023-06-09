
## MVPlot Dashboard

The MVPlot dashboard is a visualization dashboard built using Python's Dash framework. It explores NBA MVP trends from 1981 to 2021. The code is written in Python and utilizes various libraries and modules for data manipulation, visualization, and building the web application.



### Running the Dashboard

The code runs the Dash app server, allowing the MVPlot dashboard to be accessed and explored in a web browser.

If you'd like to build the application locally, it only takes two steps!

**Step One:** Install requirements.txt

```
pip install -r requirements.txt
```

**Step Two:** Build the app with Python3 🐍

```
python3 app.py
```



### The Application

**Home Page**
<img width="1269" alt="Screen Shot 2023-05-15 at 13 04 09" src="https://github.com/NoamYakar23/mvpPlot/assets/75957506/2bad2086-3039-4d85-9c3f-97548da0c745">

**Visualization Page**

<img width="1255" alt="Screen Shot 2023-05-15 at 13 03 59" src="https://github.com/NoamYakar23/mvpPlot/assets/75957506/b28d5343-e1fd-4268-b7f6-f8cef5154860">


**Statistical Modeling**

<img width="1272" alt="Screen Shot 2023-05-15 at 13 03 37" src="https://github.com/NoamYakar23/mvpPlot/assets/75957506/79092d36-9f58-4d23-8c3a-6d2b557f2bb4">



### Libraries and Modules Used

The code imports the following libraries and modules:

- `Dash`, `html`, `dcc`, `Input`, `Output` from the `dash` module for building the dashboard interface.
- `dash_table` from the `dash` module for displaying tables.
- `make_subplots` from the `plotly.subplots` module for creating subplots.
- `go` from the `plotly.graph_objects` module for creating graph objects.
- `dbc` from the `dash_bootstrap_components` module for Bootstrap components.
- `data` from the `plotly.express` module for accessing sample datasets.
- `sm` from the `statsmodels.api` module for statistical modeling.
- `px` from the `plotly.express` module for simplified plotting.
- `roc_curve`, `auc`, `f1_score` from the `sklearn.metrics` module for evaluation metrics.
- `sklearn` for general machine learning functionality.
- `pandas` for data manipulation and analysis.
- `Image` from the `PIL` (Python Imaging Library) module for working with images.
- `string` for string operations.
- `time` for time-related operations.
- `numpy` for numerical operations.
- `pickle` for object serialization.

### Data Loading and Preparation

The code loads and prepares the following datasets:

- `mvpVotingCSV`: MVP voting history data loaded from the "MVPVotingHistory.csv" file.
- `predCsv`: MVP voting data with predictions loaded from the "MvpVotingUpdatedJan14.csv" file.
- `seasonStatsDf`: Season statistics data loaded from the "seasonStatsATY.csv" file.
- `Standings`: NBA team standings data loaded from the "UpdatedStandings.csv" file.
- `topPerformers`: Top performers data loaded from the "Top Performers.csv" file.

The code also performs some data preprocessing steps, such as dropping unnecessary columns and modifying the "Team" column in the `Standings` DataFrame.

### Dashboard Setup

The code sets up the MVPlot dashboard using the following steps:

1. Creates a Dash app instance with the "COSMO" external CSS theme.
2. Assigns the Dash app server to the `server` variable.
3. Defines the styles for the sidebar and main content areas.
4. Creates the sidebar layout with a navigation menu.
5. Defines the image path for the MVP trophy image.
6. Creates the main content layout with a title, description, and image.





