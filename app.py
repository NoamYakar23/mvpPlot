from dash import Dash, html, dcc, Input, Output
import dash
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from plotly.express import data
import statsmodels.api as sm
import plotly.express as px
import pandas as pd
from PIL import Image
import string
import time

mvpVotingCSV = pd.read_csv("Datasets/MVPVotingHistory.csv")
seasonStatsDf = pd.read_csv("Datasets/seasonStatsATY.csv")

#Add winners column to csv
winners = [1 if i % 10 == 0 else 0 for i in range(0, 410)]
mvpVotingCSV['MVP'] = winners


mvpVotingCSV = mvpVotingCSV.drop(columns=['1', '2', '3','Unnamed: 0', 'Pts Max', 'Pts Won', 'First'])
Standings = pd.read_csv("Datasets/UpdatedStandings.csv")
Standings["Team"] = Standings["Team"].str.lstrip(string.digits)

topPerformers = pd.read_csv("Datasets/Top Performers.csv")
app = Dash(external_stylesheets=[dbc.themes.COSMO])
server = app.server

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


sidebar = html.Div(
    [
        html.H2("MVPlot", className="display-4"),
        html.Hr(),
        html.P(
            "A visualization dashboard that explores NBA MVP trends from 1981 - 2021",
            className="lead",
        ),
        dbc.Nav(
            [
                dbc.NavLink("The Data", href="/", active="exact"),
                dbc.NavLink("Winners and Losers", href="/winLos", active="exact"),
                dbc.NavLink("Modeling", href="/rtnModeling", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


image_path = "./mvpTrophy.png"


theData = (
    html.Div(
        children=[
            sidebar,
            html.Div(
                children=[
                    # Title of the Dashboard
                    html.H1(
                        "The Data",
                        style={"text-align": "center"},
                        className="display-3",
                    ),
                    html.H6(
                        "Inspiration, Data Scraping, and Data Cleaning",
                        className="display-6",
                        style={"text-align": "center"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Img(
                                        src=Image.open(image_path),
                                        style={
                                            "display": "inline-block",
                                            "height": "200px",
                                        },
                                    )
                                ]
                            ),
                            html.Div(
                                [
                                    dcc.Markdown(
                                        """ \
                  I'm a huge NBA fan, and obviously quite interested in analytics, so as I heard \
                  the famous *NBA on TNT Crew* talking about how Jayson Tatum could be the MVP of the league, I wondered... \
                  could I build a model to predict the next NBA MVP?
                  
                  What do we need to build this predictive model?
                 
                    * Data Source: Basketball-Reference.com
                    * Viable Method for Retrieval: Urllib & BeautifulSoup :)
                    * Period of Time for Collected Data: 1981 - 2021
                    * Insight on Relationships Between Vars: Visualization with Plotly
                """
                                    )
                                ],
                                style={
                                    "display": "inline-block",
                                    "height": "200px",
                                    "margin-left": "50px",
                                },
                            ),
                        ],
                        style={"display": "flex"},
                    ),
                ]
            ),
        ],
        style=CONTENT_STYLE,
    ),
)


winnersLosers = html.Div(
    children=[
        sidebar,
        html.Div(
            children=[
                # Title of the Dashboard
                html.H1(
                    id="mvpTitle", className="display-3", style={"text-align": "center"}
                ),
                dbc.Row(),
                html.H6("", style={"text-align": "center"}),
                html.Div(
                    [
                        html.Div(
                            id="mvpImage",
                            style={"display": "inline-block", "height": "70px"},
                        ),
                        html.Div(
                            [
                                html.H3(
                                    id="description",
                                    style={
                                        "display": "inline-block",
                                        "height": "70px",
                                        "margin-left": "50px",
                                    },
                                ),
                                html.H3(
                                    id="mvpAverages",
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "50px",
                                        "align": "center",
                                    },
                                ),
                                html.H6(
                                    id="topPerformersChampion",
                                    style={
                                        "margin-bottom": "10px",
                                        "margin-left": "50px",
                                    },
                                ),
                                html.H6(
                                    id="topPerformersRookie",
                                    style={
                                        "margin-bottom": "10px",
                                        "margin-left": "50px",
                                    },
                                ),
                                html.H6(
                                    id="topPerformersPoints",
                                    style={
                                        "margin-bottom": "10px",
                                        "margin-left": "50px",
                                    },
                                ),
                                html.H6(
                                    id="topPerformersRebounds",
                                    style={
                                        "margin-bottom": "10px",
                                        "margin-left": "50px",
                                    },
                                ),
                                html.H6(
                                    id="topPerformersAst",
                                    style={
                                        "margin-bottom": "10px",
                                        "margin-left": "50px",
                                    },
                                ),
                            ],
                            style={"align-text": "center"},
                        ),
                    ],
                    style={"display": "flex"},
                ),
                dcc.Slider(
                    1981,
                    2021,
                    1,
                    value=2020,
                    id="demo-dropdown",
                    marks={i: "{}".format("'" + str(i)[2:]) for i in range(1981, 2022)},
                ),
                dbc.Tabs(
                    [
                        dbc.Tab(label="Voting Stats", tab_id="votingStats"),
                        dbc.Tab(label="League Leaders", tab_id="leagueLeaders"),
                        dbc.Tab(label="Season Statistics", tab_id="seasonStatistics"),
                        dbc.Tab(label="Season Standings", tab_id="seasonStandings"),
                        dbc.Tab(label="Compare Players", tab_id="comparePlayers")
                    ],
                    id="tabs",
                    active_tab="scatter",
                ),
                html.Div(id="tab-content", className="p-4"),
                #                #                         dbc.Col([dcc.Graph(id = 'seasonPTSDistribution')])]),
                #                dbc.Row([dbc.Col([dcc.Graph(id = 'scoring-Dist')]), dbc.Col(dcc.Graph(id = 'fieldGoalDist'))],),
                #                dbc.Row(dcc.Graph(id = 'tableDf')))
            ],
        ),
    ],
    style=CONTENT_STYLE,
)

numericCols = [
    "Age",
    "GP",
    "MP",
    "FG%",
    "3PA",
    "3P%",
    "FT",
    "FTA",
    "ORB",
    "DRB",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
]

numeric_columns = mvpVotingCSV.select_dtypes(include=["float64", "int64"]).columns

modelList = ['Logistic Regression', 'Support Vector Machines', 'K-Nearest Neighbors']


modeling = html.Div(children = [
    sidebar,
    html.Div(
            children=[
                html.Label("Choose A Model"),
                dcc.Dropdown(
                    id="models",
                    options=[{"label": i, "value": i} for i in  modelList],
                    value="x",
                ),
            
            ],
            style=CONTENT_STYLE,
        )
    ]
                    
)

# Define the callback function to display the inputs
@app.callback(
    dash.dependencies.Output("output", "children"),
    [dash.dependencies.Input(col, "value") for col in numericCols],
)
def display_output(*args):
    return html.Div(
        [html.Div("{}".format(col)), html.Div("{:.2f}".format(val))]
        for col, val in zip(numericCols, args)
    )


content = html.Div(id="page-content")
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab")],
)
def render_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab:
        if active_tab == "votingStats":
            return html.Div(
                [
                    dcc.Graph(id="shareDist"),
                    html.H6(
                        "MVP Candidates Stats",
                        className="display-6",
                        style={"text-align": "center"},
                    ),
                    dbc.Row(dcc.Graph(id="tableDf")),
                ]
            )
        elif active_tab == "leagueLeaders":
            return html.Div(
                [
                    html.H1(
                        "League Leaders",
                        className="display-3",
                        style={"text-align": "center"},
                    ),
                    dbc.Row([dbc.Col([dcc.Graph(id="topTenners")])]),
                ]
            )
        elif active_tab == "seasonStatistics":
            return html.Div([dcc.Graph(id="seasonFG%Distribution")])

        elif active_tab == "seasonStandings":
            return html.Div(
                [
                    html.H1(
                        "Season Standings",
                        className="display-3",
                        style={"text-align": "center"},
                    ),
                    dcc.Graph(id="Standings"),
                ]
            )
        elif active_tab == "comparePlayers":
            return  html.Div(
            children=[
                html.Label("Select X axis:"),
                dcc.Dropdown(
                    id="x-axis",
                    options=[{"label": i, "value": i} for i in  numeric_columns],
                    value="x",
                ),
                html.Label("Select Y axis:"),
                dcc.Dropdown(
                    id="y-axis",
                    options=[{"label": i, "value": i} for i in numeric_columns],
                    value="y",
                ),
                dcc.Graph(id="scatterplot"),
            ],
            style={"display": "inline-block", "width": "75%"},
        )

    return "No tab selected"


# Define the callback function
@app.callback(
    dash.dependencies.Output('scatterplot', 'figure'),
    [dash.dependencies.Input('x-axis', 'value'),
     dash.dependencies.Input('y-axis', 'value'),
     Input("demo-dropdown", "value"),
     ]
)
def update_scatterplot(xaxis_column_name, yaxis_column_name, year):
    
    if year != "All":
        filtered_df = mvpVotingCSV[mvpVotingCSV["Year"] == year]
    else:
        filtered_df = mvpVotingCSV
        
    fig = go.Figure()
    
    # Add all players to the scatter plot with default marker color
    fig.add_trace(
        go.Scatter(
            x=filtered_df[xaxis_column_name],
            y=filtered_df [yaxis_column_name],
            mode='markers',
            marker_color = filtered_df["MVP"],
            text = filtered_df['Player']
        )
    )
    
    
    title = "Scatter Plot Of " +   yaxis_column_name  + " vs " +  yaxis_column_name + " During " +str(year) + "Season"

    fig.update_layout(title=title, xaxis_title=xaxis_column_name, yaxis_title=yaxis_column_name)
    return fig


@app.callback(Output("scoring-Dist", "figure"), Input("demo-dropdown", "value"))
def update_figure(selected_year):
    filtered_df = mvpVotingCSV[mvpVotingCSV["Year"] == selected_year]

    scoringPlot = px.scatter(
        filtered_df,
        x="PTS",
        y="Share",
        color="Player",
        trendline="ols",
        trendline_scope="overall",
        title="Were Volume Scorers Rewarded in the  "
        + str(selected_year)
        + " MVP Race?",
    )
    scoringPlot.update_layout(transition_duration=200)

    scoringPlot.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    return scoringPlot


@app.callback(Output("mvpTitle", "children"), Input("demo-dropdown", "value"))
def update_figure(selected_year):
    filtered_df = mvpVotingCSV[mvpVotingCSV["Year"] == selected_year]
    mvp = filtered_df.loc[filtered_df["Share"].idxmax()]["Player"]
    return f"NBA {selected_year} MVP: {mvp}"


@app.callback(Output("description", "children"), Input("demo-dropdown", "value"))
def update_figure(selected_year):
    filtered_df = mvpVotingCSV[mvpVotingCSV["Year"] == selected_year]

    mvp = filtered_df.loc[filtered_df["Share"].idxmax()]["Player"]
    lastName = mvp.split(" ")[-1]

    pts = filtered_df.loc[filtered_df["Share"].idxmax()]["PTS"]
    trb = filtered_df.loc[filtered_df["Share"].idxmax()]["TRB"]
    asts = filtered_df.loc[filtered_df["Share"].idxmax()]["AST"]

    newLine = "\n"

    ppg = f"{pts} PPG"
    rpg = f"{trb} RPG"
    apg = f" {asts} APG"
    lines = [ppg, rpg, apg]

    return f" The NBA {selected_year} MVP was {mvp}, {lastName} averaged:"


@app.callback(Output("mvpAverages", "children"), Input("demo-dropdown", "value"))
def update_figure(selected_year):
    filtered_df = mvpVotingCSV[mvpVotingCSV["Year"] == selected_year]

    mvp = filtered_df.loc[filtered_df["Share"].idxmax()]["Player"]
    lastName = mvp.split(" ")[-1]

    pts = filtered_df.loc[filtered_df["Share"].idxmax()]["PTS"]
    trb = filtered_df.loc[filtered_df["Share"].idxmax()]["TRB"]
    asts = filtered_df.loc[filtered_df["Share"].idxmax()]["AST"]

    newLine = "\n"

    ppg = f"{pts} PPG"
    rpg = f"{trb} RPG"
    apg = f" {asts} APG"
    lines = [ppg, rpg, apg]

    return " | ".join(lines)


@app.callback(Output("shareDist", "figure"), Input("demo-dropdown", "value"))
def update_figure(selected_year):

    filtered_df = mvpVotingCSV[mvpVotingCSV["Year"] == selected_year]

    shootingDist = px.histogram(
        filtered_df, x="Share", y="Player", title="Award Voting Share", orientation="h"
    )

    shootingDist.update_layout(transition_duration=200)

    return shootingDist


@app.callback(Output("mvpImage", "children"), Input("demo-dropdown", "value"))
def update_figure(selected_year):
    from PIL import Image

    filtered_df = mvpVotingCSV[mvpVotingCSV["Year"] == selected_year]
    imageName = "playerImages/" + str(selected_year) + ".jpg"

    pil_img = Image.open(imageName)
    return html.Img(src=(pil_img), style={"height": "200px", "width": "300px"})


@app.callback(
    Output("seasonFG%Distribution", "figure"), Input("demo-dropdown", "value")
)
def update_figure(selected_year):
    season_df = seasonStatsDf[seasonStatsDf["Year"] == selected_year]
    season_df = season_df.dropna()

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]],
    )

    seasonDist = go.Histogram(x=season_df["FG%"], name="FG% Distribution")
    seasonDist2 = go.Histogram(x=season_df["3P%"], name="3P% Distribution")
    seasonDist3 = go.Histogram(x=season_df["PTS"], name="PTS Distribution")

    fig.add_trace(seasonDist, row=1, col=1)
    fig.add_trace(seasonDist2, row=1, col=2)
    fig.add_trace(seasonDist3, row=1, col=3)

    fig.update_layout(
        transition_duration=200,
        title=f"Distribution of FG%, 3P%, PTS During The {selected_year} Season",
    )
    return fig


@app.callback(
    Output("season3P%Distribution", "figure"), Input("demo-dropdown", "value")
)
def update_figure(selected_year):
    season_df = seasonStatsDf[seasonStatsDf["Year"] == selected_year]
    season_df = season_df.dropna()

    seasonDist = px.histogram(
        season_df,
        x="3P%",
        title="Distribution of 3P% during the " + str(selected_year) + " Season",
    )
    seasonDist.update_layout(transition_duration=200)

    return seasonDist


def outputTopTen(df, category):

    # BOLD Text
    start = "\033[1m"
    end = "\033[0;0m"

    topTen = df.sort_values(by=category, ascending=False)[:10]
    #     print("\n\n"+ start + f"Top 10 {category} leaders in the League" + end + "\n\n")
    #     for playerName in topTen['Player'].values:
    #         print(playerName, topTen[topTen['Player'] == playerName][category].values)
    return topTen


@app.callback(Output("topTenners", "figure"), Input("demo-dropdown", "value"))
def update_figure(selected_year):
    season_df = seasonStatsDf[seasonStatsDf["Year"] == selected_year]
    season_df = season_df.dropna()

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "table"}, {"type": "table"}, {"type": "table"}]],
    )

    topTenPTS = outputTopTen(season_df, "PTS")
    topTenTRB = outputTopTen(season_df, "TRB")
    topTenAST = outputTopTen(season_df, "AST")

    fig.add_trace(
        go.Table(
            header=dict(values=("Player", "PTS"), align="left"),
            cells=dict(values=[topTenPTS.Player, topTenPTS.PTS.values], align="left"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Table(
            header=dict(values=("Player", "AST"), align="left"),
            cells=dict(values=[topTenAST.Player, topTenAST.AST], align="left"),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Table(
            header=dict(values=("Player", "TRB"), align="left"),
            cells=dict(values=[topTenTRB.Player, topTenTRB.TRB], align="left"),
        ),
        row=1,
        col=3,
    )

    return fig


@app.callback(Output("tableDf", "figure"), Input("demo-dropdown", "value"))
def update_figure(selected_year):
    filtered_df = mvpVotingCSV[mvpVotingCSV["Year"] == selected_year]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=("Player", "PTS", "AST", "STL", "TRB", "3P%", "FG%"),
                    align="left",
                ),
                cells=dict(
                    values=[
                        filtered_df.Player,
                        filtered_df.PTS,
                        filtered_df.AST,
                        filtered_df.STL,
                        filtered_df.TRB,
                        filtered_df["3P%"],
                        filtered_df["FG%"],
                    ],
                    align="left",
                ),
            )
        ]
    )
    return fig


@app.callback(Output("Standings", "figure"), Input("demo-dropdown", "value"))
def update_figure(selected_year):
    filtered_df = Standings[Standings["Year"] == selected_year]
    filtered_df = (
        filtered_df[["Team", "W", "L", "Pct"]]
        .drop_duplicates()
        .sort_values(by="W", ascending=False)
    )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=("Team", "W", "L", "Pct"), align="left"),
                cells=dict(
                    values=[
                        filtered_df.Team,
                        filtered_df.W,
                        filtered_df.L,
                        filtered_df.Pct,
                    ],
                    align="left",
                ),
            )
        ]
    )
    return fig


@app.callback(
    Output("topPerformersChampion", "children"), Input("demo-dropdown", "value")
)
def update_figure(selected_year):
    filtered_df = topPerformers[topPerformers["Season"] == selected_year]

    return f"NBA Champion: " + filtered_df["Champion"] + "🏆"


@app.callback(
    Output("topPerformersRookie", "children"), Input("demo-dropdown", "value")
)
def update_figure(selected_year):
    filtered_df = topPerformers[topPerformers["Season"] == selected_year]

    return f"Rookie of the Year: " + filtered_df["Rookie of the Year"] + "⭐"


@app.callback(
    Output("topPerformersPoints", "children"), Input("demo-dropdown", "value")
)
def update_figure(selected_year):
    filtered_df = topPerformers[topPerformers["Season"] == selected_year]

    return f"Points Leader: " + filtered_df["Points"] + "🏀"


@app.callback(
    Output("topPerformersRebounds", "children"), Input("demo-dropdown", "value")
)
def update_figure(selected_year):
    filtered_df = topPerformers[topPerformers["Season"] == selected_year]

    return f"Rebounds Leader: " + filtered_df["Rebounds"] + "🪱"


@app.callback(Output("topPerformersAst", "children"), Input("demo-dropdown", "value"))
def update_figure(selected_year):
    filtered_df = topPerformers[topPerformers["Season"] == selected_year]

    return f"Assists Leader: " + filtered_df["Assists"] + "🪙"


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return theData
    elif pathname == "/winLos":
        return winnersLosers
    elif pathname == "/rtnModeling":
        return modeling
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == "__main__":
    app.run_server(debug=True)
