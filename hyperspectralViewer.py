import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import cv2
import base64
import numpy
from dash.exceptions import PreventUpdate
from flask_caching import Cache
import flask
import matplotlib.path as mpltPath
import time
import h5py
from dash_extensions import Download
import io
import json
import zipfile
import xml.etree.cElementTree as ET
from sklearn.preprocessing import normalize,MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
import matplotlib.cm

# Data to load
resultsfile='hypix.h5'
inputFile=h5py.File(resultsfile,'r')
configFile='config.json'
f=open(configFile,'r')
cfg=json.load(f)
f.close()

wavelength=[383.26312,384.1194,384.97583,385.83243,386.6892,387.54617,388.40332,389.26062,390.1181,390.97577,391.83362,392.69162,393.54977,394.40814,395.26666,396.12537,396.98425,397.8433,398.7025,399.56192,400.42148,401.28122,402.14114,403.00122,403.86148,404.72192,405.58255,406.44333,407.3043,408.1654,409.02673,409.8882,410.74985,411.6117,412.4737,413.33588,414.1982,415.06073,415.92343,416.78632,417.64935,418.51257,419.37598,420.23953,421.10327,421.9672,422.83127,423.69553,424.55997,425.4246,426.28937,427.15433,428.01944,428.88477,429.7502,430.61588,431.48172,432.34772,433.21387,434.08023,434.94675,435.81345,436.6803,437.54736,438.41455,439.28195,440.1495,441.01724,441.88516,442.75323,443.62152,444.48993,445.35855,446.22733,447.09628,447.9654,448.83472,449.7042,450.57382,451.44366,452.31366,453.1838,454.05417,454.92468,455.79538,456.66623,457.53726,458.40848,459.27988,460.15143,461.02316,461.89508,462.76715,463.6394,464.51184,465.38443,466.2572,467.13016,468.0033,468.8766,469.75006,470.62372,471.49753,472.37152,473.2457,474.12003,474.99454,475.86923,476.74408,477.6191,478.49432,479.36972,480.24527,481.121,481.99692,482.873,483.74924,484.62567,485.50226,486.37903,487.25598,488.13312,489.0104,489.88788,490.7655,491.64334,492.5213,493.39948,494.27783,495.15634,496.035,496.91388,497.7929,498.67212,499.55148,500.43103,501.31076,502.19067,503.07074,503.951,504.83142,505.712,506.59277,507.47372,508.35486,509.23615,510.1176,510.99924,511.88107,512.76306,513.6452,514.5275,515.41003,516.2927,517.1756,518.0586,518.94183,519.8252,520.70874,521.59247,522.4764,523.3604,524.2447,525.1291,526.0137,526.89844,527.7834,528.6685,529.55383,530.4393,531.3249,532.2107,533.0967,533.98285,534.8692,535.7557,536.6424,537.52924,538.41626,539.30347,540.19086,541.07837,541.9661,542.854,543.74207,544.6303,545.51874,546.4073,547.2961,548.185,549.0741,549.9634,550.85284,551.7425,552.6323,553.5223,554.4124,555.3028,556.1933,557.0839,557.9748,558.86584,559.757,560.6484,561.5399,562.43164,563.32355,564.21564,565.10785,566.00024,566.8928,567.7856,568.6785,569.57166,570.46497,571.3584,572.252,573.1458,574.0398,574.93396,575.8283,576.7228,577.61743,578.51227,579.4073,580.3025,581.1979,582.09344,582.98914,583.885,584.78107,585.6773,586.5737,587.47034,588.36707,589.26404,590.16113,591.0584,591.9559,592.85345,593.7513,594.64923,595.5474,596.44574,597.34424,598.24286,599.1417,600.0408,600.93994,601.8393,602.73883,603.63855,604.53845,605.4385,606.33875,607.23914,608.1397,609.04047,609.9414,610.8425,611.7438,612.6452,613.5469,614.44867,615.35065,616.25275,617.1551,618.05756,618.96027,619.8631,620.7661,621.6693,622.57263,623.4762,624.3799,625.28375,626.1878,627.09204,627.99646,628.90106,629.8058,630.7107,631.61584,632.5211,633.4265,634.33215,635.238,636.1439,637.05005,637.95636,638.86285,639.76953,640.6764,641.5834,642.49054,643.39795,644.3055,645.21313,646.12103,647.0291,647.9373,648.8457,649.7543,650.663,651.57196,652.4811,653.3903,654.2998,655.2094,656.1192,657.0292,657.9393,658.8496,659.7601,660.6708,661.5816,662.4926,663.40375,664.3151,665.2267,666.13837,667.05023,667.9623,668.8745,669.7869,670.69946,671.61224,672.52515,673.43823,674.3515,675.26495,676.1786,677.09235,678.0063,678.9205,679.8348,680.74927,681.6639,682.57874,683.4937,684.40894,685.3243,686.2398,687.1555,688.07135,688.9874,689.9036,690.82,691.7366,692.6533,693.57025,694.48737,695.4046,696.322,697.2397,698.1575,699.0754,699.9935,700.91187,701.8303,702.74896,703.6678,704.5868,705.506,706.42535,707.34485,708.2645,709.18445,710.1045,711.02466,711.94507,712.86566,713.7864,714.7073,715.62836,716.5496,717.47107,718.3927,719.31445,720.23645,721.15857,722.0809,723.00336,723.92596,724.8488,725.7718,726.695,727.61835,728.5419,729.4655,730.3894,731.3135,732.2377,733.16205,734.0866,735.01135,735.9363,736.8613,737.7866,738.71204,739.63763,740.5634,741.4894,742.41547,743.3418,744.26825,745.19495,746.12177,747.0487,747.9759,748.90326,749.83075,750.7585,751.68634,752.6144,753.54254,754.47095,755.39954,756.32825,757.25714,758.1862,759.1155,760.0449,760.97455,761.9043,762.8342,763.76434,764.69464,765.6251,766.5558,767.4866,768.4176,769.34875,770.2801,771.2116,772.1433,773.07513,774.0072,774.9394,775.87177,776.8043,777.73706,778.67,779.603,780.5363,781.4697,782.4033]

# Server Setup with caching
server = flask.Flask(__name__)
app = dash.Dash(__name__,server=server,external_stylesheets=[dbc.themes.COSMO])#dbc.themes.GRID #COSMO #LUX
app.title="Hyperspectral Viewer"
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR':'./cache05',
    'CACHE_THRESHOLD':300,
    'CACHE_DEFAULT_TIMEOUT':0
})

#global variables
colors=['#636EFA','#FDF150','#EF553B','#03CC96']
rgbcolors=[(250,110,99),(10,241,255),(59,85,239),(150,204,3)]
# rgbcolors=[(64000,28160,25344),(2560,61696,65280),(15104,21760,61184),(38400,52224,768)]
timepoints=['T0','T0.5','T1','T2']
SPECTRAL_DIM=447

# HTML layout
app.layout = dbc.Container(fluid=True,children=[
    dcc.Store(id="proxy"),
    dbc.Row([
        dbc.Col(html.Div([
            dbc.Label("Contamination", html_for="contamination"),
            dbc.RadioItems(
                id='contamination',
                options=[{'label': i, 'value': i} for i in ['control','barite','bento','drill_cutting']],
                value='control',
                # labelStyle={'display': 'inline-block'}
                inline=True
            ),
        ])),
        dbc.Col(html.Div([
            dbc.Label('Concentration',html_for="concentration"),
            dbc.RadioItems(
                id='concentration',
                options=[{'label': i, 'value': i} for i in ["bentobarite","drill_cutting"]],
                value='bentobarite',
                # labelStyle={'display': 'inline-block'}
                inline=True
            ),
        ])),
        dbc.Col(html.Div([
            dbc.Label('Dimension Reduction',html_for="dimredtechnique"),
            dbc.RadioItems(
                id='dimredtechnique',
                options=[{'label': i, 'value': i} for i in ['t-SNE', 'UMAP','UMAPcos']],
                value='t-SNE',
                inline=True
                #labelStyle={'display': 'inline-block'}
            ),
        ])),
        dbc.Col(html.Div([
            dbc.Label('Coral',html_for="coral"),
            dcc.Slider(
                id='coral',
                min=0,
                max=5,
                value=0,
                marks={i: i for i in ['0','1','2','3','4','5']},
                step=None,
            )
        ])),
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col(html.Div(className="loader-wrapper",children=[
            dcc.Loading(
                id="load",
                type="cube",
                style={"-moz-transform": "scale(2)","-ms-transform": "scale(2)","-o-transform": "scale(2)","-webkit-transform": "scale(2)","transform": "scale(2)"},
                children=[
                dcc.Graph(
                    id='scatter',
                    style={'height': "85vh"},
                    clear_on_unhover=True,
                    config = {'doubleClickDelay': 500,
                    'modeBarButtonsToRemove':[
                        'hoverCompareCartesian'
                    ]},
                )
            ]),
        ]),width=7),

        dbc.Col(html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    dbc.Label('Aggregation Operation',html_for='aggop'),
                    dbc.RadioItems(
                        id='aggop',
                        options=[{'label': i, 'value': i,"label_id":str(i)+"lbl"} for i in ['mean','max','min','median']],
                        value='mean',
                        inline=True
                    )
                ])),
                dbc.Tooltip("Use mean as aggregation operation for the images and spectra.",
                    target="meanlbl",
                ),
                dbc.Tooltip("Use max as aggregation operation for the images and spectra.",
                    target="maxlbl",
                ),
                dbc.Tooltip("Use min as aggregation operation for the images and spectra.",
                    target="minlbl",
                ),
                dbc.Tooltip("Use median as aggregation operation for the images and spectra.",
                    target="medianlbl",
                ),
                dbc.Col(html.Div([
                    dbc.Label('Normalization',html_for='norm'),
                    dbc.Checklist(
                        options=[
                            {"label": "Normalize Images", "value": 1,"label_id":"normImgs"},
                            {"label": "Pseudocolor", "value": 2,"label_id":"pseudocolor"},
                            {"label": "Normalize Spectra", "value": 3,"label_id":"normSpectra"},
                        ],
                        value=[3],
                        inline=True,
                        id="norm",
                        switch=True,
                    ),
                    dbc.Tooltip("Normalize Images to [0,255]. Ignores Brightness.",
                        target="normImgs",
                    ),
                    dbc.Tooltip("Applies pseudocolor mapping. Ignores Normalize Images and Brightness.",
                        target="pseudocolor",
                    ),
                    dbc.Tooltip("Normalize Spectra using l1-norm. Also applies to data for PCA biplot.",
                        target="normSpectra",
                    ),
                    
                    
                ])),
            ]),
            html.Hr(),
            dbc.Row([
                dcc.Graph(id='img0',config={
                    'modeBarButtonsToRemove': [
                        'sendDataToCloud',
                        'editInChartStudio',
                        'zoomIn2d',
                        'zoomOut2d',
                        'autoScale2d',
                        'resetScale2d',
                        'hoverCompareCartesian',
                        'hoverClosestCartesian',
                        'toggleSpikelines',
                        'toImage',
                    ],
                    'doubleClickDelay': 500,
                    'displaylogo': False
                }),
                dcc.Graph(id='img05',config={
                    'modeBarButtonsToRemove': [
                        'sendDataToCloud',
                        'editInChartStudio',
                        'zoomIn2d',
                        'zoomOut2d',
                        'autoScale2d',
                        'resetScale2d',
                        'hoverCompareCartesian',
                        'hoverClosestCartesian',
                        'toggleSpikelines',
                        'toImage'
                    ],
                    'doubleClickDelay': 500,
                    'displaylogo': False
                }),
                dcc.Graph(id='img1',config={
                    'modeBarButtonsToRemove': [
                        'sendDataToCloud',
                        'editInChartStudio',
                        'zoomIn2d',
                        'zoomOut2d',
                        'autoScale2d',
                        'resetScale2d',
                        'hoverCompareCartesian',
                        'hoverClosestCartesian',
                        'toggleSpikelines',
                        'toImage'
                    ],
                    'doubleClickDelay': 500,
                    'displaylogo': False
                }),
                dcc.Graph(id='img2',config={
                    'modeBarButtonsToRemove': [
                        'sendDataToCloud',
                        'editInChartStudio',
                        'zoomIn2d',
                        'zoomOut2d',
                        'autoScale2d',
                        'resetScale2d',
                        'hoverCompareCartesian',
                        'hoverClosestCartesian',
                        'toggleSpikelines',
                        'toImage'
                    ],
                    'doubleClickDelay': 500,
                    'displaylogo': False
                }),
            ]),
            dbc.Row([
                dbc.Col(dbc.Label("Brightness", html_for="brightness",id="brightnesslbl"),
                width=2),
                dbc.Col(dcc.Slider(
                    id='brightness',
                    min=1,
                    max=10,
                    value=1,
                    marks={str(i): str(i) for i in range(1,11)},
                    step=None
                ))
            ]),
            dbc.Tooltip("Increase the brightness of the displayed images by the selected factor.",
                target="brightnesslbl",
            ),
            dbc.Row([
                dbc.Col(dbc.Label("Channel", html_for="channel",id="channellbl"),
                width=2),
                dbc.Col(dcc.Slider(
                    id='channel',
                    min=-1,
                    max=SPECTRAL_DIM-1,
                    value=-1,
                    tooltip={},
                    marks={-1:"all","100":str(wavelength[100]),"200":str(wavelength[200]),"300":str(wavelength[300]),"400":str(wavelength[400])},
                )),
                dbc.Tooltip("Display the selected channel as image series.",
                        target="channellbl",
                )
            ]),        
            html.Hr(),
            dbc.Row([
                dbc.Col(dcc.Graph(
                        id='spectra',
                ))
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col(html.Div([
                    dbc.Button("Download Selection", color="info", className="mr-1",id="download"),
                    dbc.Tooltip("Download scatter plot data selection as Envie compatible xml file.",
                        target="download",
                    ),
                    dbc.Button("Show PCA biplot", color="info", className="mr-1",id="showBiplot"),
                    dbc.Tooltip("Creates PCA biplot of (selected) data. Only (re-)created on click.",
                        target="showBiplot",
                    ),
                    Download(id="downloadhelper"),
                    
                ]))
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(
                        id='biplot',
                ))
            ])
        ])),
    ]),
])


app.clientside_callback(
    """
        function HoverGate(selected,hovered) {
            if(selected){
                console.log("You shall not pass")
                return window.dash_clientside.no_update;
            }else{
                //enable throtteling
                let t_throttle = 250;
                // ns refers to the clientside namespace
                ns.t0 = ns.t0 || 0;
                let dt = Date.now() - ns.t0;
                if ( dt < t_throttle) {
                    return window.dash_clientside.no_update
                } else {
                    ns.t0 = Date.now();
                    return [{"changed":"True"}];
                }
            }
        }
    """,
    dash.dependencies.Output('proxy', 'data'),
    [dash.dependencies.Input('scatter', 'selectedData'),
    dash.dependencies.Input('scatter', 'hoverData')])

@app.callback(
    [dash.dependencies.Output('img0', 'selectedData'),
    dash.dependencies.Output('img05', 'selectedData'),
    dash.dependencies.Output('img1', 'selectedData'),
    dash.dependencies.Output('img2', 'selectedData'),
    dash.dependencies.Output('scatter', 'selectedData'),
    dash.dependencies.Output('scatter', 'hoverData')],
    [dash.dependencies.Input('dimredtechnique', 'value'),
     dash.dependencies.Input('contamination', 'value'),
     dash.dependencies.Input('concentration', 'value'),
     dash.dependencies.Input('coral', 'value')])
def InputChanged(dimredtechnique_val,contamination_val,concentration_val,coral_val):
    return [None,None,None,None,None,None]

#Create main scatter plot
@app.callback(
    dash.dependencies.Output('scatter', 'figure'),
    [dash.dependencies.Input('dimredtechnique', 'value'),
     dash.dependencies.Input('contamination', 'value'),
     dash.dependencies.Input('concentration', 'value'),
     dash.dependencies.Input('coral', 'value'),
     dash.dependencies.Input('img0','selectedData'),
     dash.dependencies.Input('img05','selectedData'),
     dash.dependencies.Input('img1','selectedData'),
     dash.dependencies.Input('img2','selectedData')])
@cache.memoize()
def update_scatter(dimredtechnique_val,contamination_val,concentration_val,coral_val,img0_lasso,img05_lasso,img1_lasso,img2_lasso):
    if dash.callback_context.triggered[0]['prop_id'] in ['contamination.value','coral.value','concentration.value','dimredtechnique.value']:
        img0_lasso=None
        img05_lasso=None
        img1_lasso=None
        img2_lasso=None
    if checkInvalidSelect(concentration_val,contamination_val):
        raise PreventUpdate
    graphs=[]
    lassos=[img0_lasso, img05_lasso, img1_lasso,img2_lasso]
    for idx,t in enumerate(timepoints):
        data=inputFile[dimredtechnique_val][contamination_val][str(concentration_val)][str(coral_val)][t]
        if lassos[idx]:
            if 'lassoPoints' in lassos[idx].keys():
                imgsel=mpltPath.Path(list(zip(lassos[idx]['lassoPoints']['x'],lassos[idx]['lassoPoints']['y']))).contains_points(list(zip(data[:,2].astype(int),data[:,3].astype(int))))
            #otherwise it must be box selection
            else:
                imgsel=mpltPath.Path([(lassos[idx]['range']['x'][0], lassos[idx]['range']['y'][0]),(lassos[idx]['range']['x'][0], lassos[idx]['range']['y'][1]),(lassos[idx]['range']['x'][1], lassos[idx]['range']['y'][1]),(lassos[idx]['range']['x'][1], lassos[idx]['range']['y'][0])]).contains_points(list(zip(data[:,2].astype(int),data[:,3].astype(int))))
            data=numpy.array(data)
            graphs.append(dict(
            type="scattergl",
            y=data[imgsel,1],
            x=data[imgsel,0],
            customdata=list(zip(data[imgsel,2].astype(int),data[imgsel,3].astype(int))),
            mode="markers",
            name=t,
            marker={
                "color":colors[idx],
            }
            ))
        else:
            graphs.append(dict(
            type="scattergl",
            y=data[:,1],
            x=data[:,0],
            customdata=list(zip(data[:,2].astype(int),data[:,3].astype(int))),
            mode="markers",
            name=t,
            marker={
                "color":colors[idx],
            }
            ))
    return dict(data=graphs,layout={'dragmode': 'lasso',"hovermode":"closest",'plot_bgcolor':'rgb(40,40,40)','margin':{'l':17,'r':5,'t':17,'b':25}})


# helperFunction to create scatterPlot image
def createImageFig(img):
    fig = go.Figure()

    # Constants
    img_height = img.shape[0]
    img_width = img.shape[1]
    scale_factor = 1

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )
    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )
    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x",
    )
    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source='data:image/png;base64,{}'.format(base64.b64encode(cv2.imencode('.png',img)[1]).decode("utf-8")))
    )
    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 5, "b": 5},
        dragmode='lasso'
    )
    return fig


def checkInvalidSelect(concentration_val,contamination_val):
    try:
        inputFile["/metadata/"+str(contamination_val)+"/"+str(concentration_val)]
        return False
    except KeyError:
        return True


def myNormData(tmpdata,normalizeData):
    if normalizeData:
        return normalize(tmpdata.reshape(-1,447),'l1').reshape(tmpdata.shape)
    return tmpdata


@app.callback(
    dash.dependencies.Output('spectra', 'figure'),
    [dash.dependencies.Input('dimredtechnique', 'value'),
    dash.dependencies.Input('contamination', 'value'),
    dash.dependencies.Input('concentration', 'value'),
    dash.dependencies.Input('coral', 'value'),
    dash.dependencies.Input('aggop','value'),
    dash.dependencies.Input('scatter', 'selectedData'),
    dash.dependencies.Input('norm','value'),
    dash.dependencies.Input('img0','selectedData'),
    dash.dependencies.Input('img05','selectedData'),
    dash.dependencies.Input('img1','selectedData'),
    dash.dependencies.Input('img2','selectedData'),
    dash.dependencies.Input('proxy','data')],
    dash.dependencies.State('scatter','hoverData')
    )
def update_spectra(dimredtechnique_val,contamination_val,concentration_val,coral_val,aggop_val,selectedData,normalizeData,img0_lasso,img05_lasso,img1_lasso,img2_lasso,proxydummy,hoverData):
    #if data has changed set selection to None, should be unessecary but reset function is run after this function executed
    if dash.callback_context.triggered[0]['prop_id'] in ['contamination.value','coral.value','concentration.value','dimredtechnique.value']:
        selectedData=None
        hoverData=None

    if checkInvalidSelect(concentration_val,contamination_val):
        raise PreventUpdate

    data=None
    lassos=[img0_lasso, img05_lasso, img1_lasso,img2_lasso]

    normalizeData = 3 in normalizeData
    myaggOperator=numpy.mean
    if aggop_val=='max':
        myaggOperator=numpy.max
    elif aggop_val=='min':
        myaggOperator=numpy.min
    elif aggop_val=='median':
        myaggOperator=numpy.median
    meanSpectra=numpy.zeros((4,SPECTRAL_DIM))
    if selectedData:
        #make selection linear
        sel=[[],[],[],[]]
        for i in selectedData['points']:
            sel[i['curveNumber']].append(i['customdata'])
        for idx,s in enumerate(sel):
            if s:
                nsel=numpy.array(s)
                #aggregate the selection
                curData=myNormData(inputFile["/metadata/"+str(contamination_val)+"/"+str(concentration_val)+"/"+str(coral_val)+"/"+timepoints[idx]+"/hyperspectralimage"][()][nsel[:,1],nsel[:,0]],normalizeData)
                meanSpectra[idx]=myaggOperator(curData,0)
    elif hoverData:
        y,x=hoverData['points'][0]['customdata']
        curve=hoverData['points'][0]['curveNumber']
        data=myNormData(inputFile["/metadata/"+str(contamination_val)+"/"+str(concentration_val)+"/"+str(coral_val)+"/"+timepoints[curve]+"/hyperspectralimage"][()][x,y],normalizeData)
        meanSpectra[curve]=data
    elif numpy.any(list(map(numpy.any,lassos))):
        for idx,t in enumerate(timepoints):
            data=inputFile[dimredtechnique_val][contamination_val][str(concentration_val)][str(coral_val)][t]
            if lassos[idx]:
                if 'lassoPoints' in lassos[idx].keys():
                    imgsel=mpltPath.Path(list(zip(lassos[idx]['lassoPoints']['x'],lassos[idx]['lassoPoints']['y']))).contains_points(list(zip(data[:,2].astype(int),data[:,3].astype(int))))
                #otherwise it must be box selection
                else:
                    imgsel=mpltPath.Path([(lassos[idx]['range']['x'][0], lassos[idx]['range']['y'][0]),(lassos[idx]['range']['x'][0], lassos[idx]['range']['y'][1]),(lassos[idx]['range']['x'][1], lassos[idx]['range']['y'][1]),(lassos[idx]['range']['x'][1], lassos[idx]['range']['y'][0])]).contains_points(list(zip(data[:,2].astype(int),data[:,3].astype(int))))
                data=numpy.array(data)
                x,y=data[imgsel,2].astype(int),data[imgsel,3].astype(int)
                curData=myNormData(inputFile["/metadata/"+str(contamination_val)+"/"+str(concentration_val)+"/"+str(coral_val)+"/"+timepoints[idx]+"/hyperspectralimage"][()][x,y],normalizeData)
                meanSpectra[idx]=myaggOperator(curData,0)
            else:
                allsel=inputFile["UMAPcos"][contamination_val][str(concentration_val)][str(coral_val)][timepoints[idx]][:,2:].astype(int)
                curData=myNormData(inputFile["/metadata/"+str(contamination_val)+"/"+str(concentration_val)+"/"+str(coral_val)+"/"+timepoints[idx]+"/hyperspectralimage"][()][allsel[:,1],allsel[:,0]],normalizeData)
                meanSpectra[idx]=myaggOperator(curData,0)
    else:
        for idx in range(4):
            allsel=inputFile["UMAPcos"][contamination_val][str(concentration_val)][str(coral_val)][timepoints[idx]][:,2:].astype(int)
            curData=myNormData(inputFile["/metadata/"+str(contamination_val)+"/"+str(concentration_val)+"/"+str(coral_val)+"/"+timepoints[idx]+"/hyperspectralimage"][()][allsel[:,1],allsel[:,0]],normalizeData)
            meanSpectra[idx]=myaggOperator(curData,0)

    
    #add one or more spectra to plot
    data=[]
    for idx,(t,c) in enumerate(zip(timepoints,colors)):
        if meanSpectra[idx].sum():
            data.append(dict(
                type="scattergl",
                y=meanSpectra[idx],
                x=wavelength,#numpy.arange(383,383+SPECTRAL_DIM),
                mode="line",
                name=t,
                marker={
                    "color":c,
                }
            ))
    return dict(data=data,layout={'showlegend':True,'plot_bgcolor':'rgb(40,40,40)','margin':{'l':45,'r':5,'t':17,'b':25}})

def myImgNorm(img,norm,brightness):
    if 2 in norm:
        m=MinMaxScaler((0,1))
        tmp=(matplotlib.cm.Spectral(m.fit_transform(img))*255).astype(numpy.uint8)
        return tmp
    if 1 in norm:
        m=MinMaxScaler((0,255))
        return m.fit_transform(img).astype(numpy.uint8)
    return (img/65536*255*10*brightness).astype(numpy.uint8)


@app.callback(
    [dash.dependencies.Output('img0', 'figure'),
    dash.dependencies.Output('img05', 'figure'),
    dash.dependencies.Output('img1', 'figure'),
    dash.dependencies.Output('img2', 'figure')],
    [dash.dependencies.Input('dimredtechnique', 'value'),
     dash.dependencies.Input('contamination', 'value'),
     dash.dependencies.Input('concentration', 'value'),
     dash.dependencies.Input('coral', 'value'),
     dash.dependencies.Input('brightness', 'value'),
     dash.dependencies.Input('scatter', 'selectedData'),
     dash.dependencies.Input('channel','value'),
     dash.dependencies.Input('aggop','value'),
     dash.dependencies.Input('norm','value')]
     )
def update_images(dimredtechnique_val,contamination_val,concentration_val,coral_val,brightness_val,selectedData,channel,aggop_val,norm_val):
    if checkInvalidSelect(concentration_val,contamination_val):
        raise PreventUpdate
    imgs=[]

    if selectedData and (2 in norm_val):
        norm_val.remove(2)
    
    #monkey patching ftw
    myaggOperator=numpy.mean
    if aggop_val=='max':
        myaggOperator=numpy.max
    elif aggop_val=='min':
        myaggOperator=numpy.min
    elif aggop_val=='median':
        myaggOperator=numpy.median
    
    #load and manipulate images
    if channel==-1:
        for timepoint in timepoints:
            #saved img
            # imgs.append(inputFile['/metadata/'+contamination_val+'/'+str(concentration_val)+"/"+str(coral_val)+"/"+timepoint+"/image"][()]*brightness_val)
            #live img
            imgs.append(myImgNorm(myaggOperator(inputFile['/metadata/'+contamination_val+'/'+str(concentration_val)+"/"+str(coral_val)+"/"+timepoint+"/hyperspectralimage"][()],2),norm_val,brightness_val))
    else:
        for timepoint in timepoints:
            #live img
            imgs.append(myImgNorm(inputFile['/metadata/'+contamination_val+'/'+str(concentration_val)+"/"+str(coral_val)+"/"+timepoint+"/hyperspectralimage"][:,:,channel],norm_val,brightness_val))
    
    
    if selectedData:
        #make selection linear
        sel=[[],[],[],[]]
        for i in selectedData['points']:
            sel[i['curveNumber']].append(i['customdata'])
        for idx,s in enumerate(sel):
            nsel=numpy.array(s)
            #paint the selected parts in the image
            imgs[idx]=cv2.cvtColor(imgs[idx],cv2.COLOR_GRAY2RGB)#numpy.dstack((imgs[idx],imgs[idx],imgs[idx])) #convert gray to rgb
            if nsel.size:
                imgs[idx][(nsel[:,1],nsel[:,0])]=rgbcolors[idx]
    imgdata=[createImageFig(cv2.flip(img,0)) for img in imgs]
    
    return imgdata

@app.callback(
    [dash.dependencies.Output('coral', 'min'),
    dash.dependencies.Output('coral', 'max'),
    dash.dependencies.Output('coral', 'value')],
    [dash.dependencies.Input('contamination', 'value'),
    dash.dependencies.Input('concentration', 'value')])
def update_num_corals(cont,conc):
    if cont=='drill_cutting' or (cont=='control' and conc=='drill_cutting'):
        return 0,5,0
    return 0,4,0

@app.callback(
    [dash.dependencies.Output('concentration', 'options'),
    dash.dependencies.Output('concentration', 'value')],
    [dash.dependencies.Input('contamination', 'value')])
def update_impossible_configurations(cont):
    options=cfg[cont].keys()
    return [{'label': i, 'value': i} for i in options],list(options)[0]

def getCoralName(contamination_val,concentration_val,coral_val):
    retval=""
    if contamination_val=="drill_cutting":
        retval="coral"
    else:
        retval=contamination_val[:2]
    if contamination_val!="control":
        retval+=str(concentration_val+coral_val+1) #TODO this is problematic because of the +1
    else:
        retval+=str(coral_val+1)
    return retval

@app.callback(
    dash.dependencies.Output('downloadhelper','data'),
    [dash.dependencies.Input('download', 'n_clicks')],
    [dash.dependencies.State('dimredtechnique', 'value'),
     dash.dependencies.State('contamination', 'value'),
     dash.dependencies.State('concentration', 'value'),
     dash.dependencies.State('coral', 'value'),
     dash.dependencies.State('scatter', 'selectedData')])
def download_data(n_clicks, dimredtechnique_val,contamination_val,concentration_val,coral_val,selectedData):
    if selectedData:
        out=io.BytesIO()
        zipObj = zipfile.ZipFile(out, 'w')
        for idx,t in enumerate(cfg[contamination_val][concentration_val]):
            coordstr=""
            for i in selectedData['points']:
                imgIdx=i['curveNumber']
                if imgIdx==idx:        
                    y=inputFile["/metadata/"+str(contamination_val)+"/"+str(concentration_val)+"/"+str(coral_val)+"/"+t+"/min/y"][()]+i['customdata'][1]
                    x=inputFile["/metadata/"+str(contamination_val)+"/"+str(concentration_val)+"/"+str(coral_val)+"/"+t+"/min/x"][()]+i['customdata'][1]
                    coordstr+=str(x)+" "+str(y)+" "
            rois = ET.Element("RegionsOfInterest",version="1.1")
            roi = ET.SubElement(rois, "Region",name=getCoralName(contamination_val,concentration_val,coral_val),color="128,0,0")
            pdef = ET.SubElement(roi, "PixelDef")
            ET.SubElement(pdef, "SpatialRef").text="none"
            ET.SubElement(pdef, "Coordinates").text=coordstr
            tree = ET.ElementTree(rois)
            tmpToBeZippedFile=io.BytesIO()
            tree.write(tmpToBeZippedFile,encoding="UTF-8", xml_declaration=True)
            zipObj.writestr(inputFile["/metadata/"+str(contamination_val)+"/"+str(concentration_val)+"/"+str(coral_val)+"/"+t+"/file"][()].decode('utf-8').split("/")[-1].replace('.h5','.xml'),tmpToBeZippedFile.getvalue())
            tmpToBeZippedFile.close()
        zipObj.close()
        retval=base64.b64encode(out.getvalue())
        out.close()
        return dict(content=retval.decode(), filename=str(dimredtechnique_val)+"_"+str(contamination_val)+"_"+str(concentration_val)+"_"+str(coral_val)+str(time.time())+".zip", mime_type='application/zip', base64=True)
    return None

def nm2Color(nm):
    ALPHA='0.65'
    if nm<=450:
        return 'rgba(121,121,206,'+ALPHA+')'
    if nm>450 and nm<=500:
        return 'rgba(121,185,249,'+ALPHA+')'
    if nm>500 and nm<=550:
        return 'rgba(124,249,248,'+ALPHA+')'
    if nm>550 and nm<=600:
        return 'rgba(186,249,185,'+ALPHA+')'
    if nm>600 and nm<=650:
        return 'rgba(249,248,122,'+ALPHA+')'
    if nm>650 and nm<=700:
        return 'rgba(249,185,121,'+ALPHA+')'
    if nm>700:
        return 'rgba(205,121,121,'+ALPHA+')'


@app.callback(
    [dash.dependencies.Output('biplot','figure'),
    dash.dependencies.Output('biplot','style')],
    [dash.dependencies.Input('showBiplot', 'n_clicks'),
     dash.dependencies.Input('contamination', 'value'),
     dash.dependencies.Input('concentration', 'value'),
     dash.dependencies.Input('coral', 'value'),
     dash.dependencies.Input('scatter', 'selectedData'),
     dash.dependencies.Input('norm','value')])
def createBiplot(n_clicks,contamination_val,concentration_val,coral_val,selectedData,normalizeData):
    if dash.callback_context.triggered[0]['prop_id'] != 'showBiplot.n_clicks':
        return dict(data=None,layout={'plot_bgcolor':'rgb(40,40,40)','margin':{'l':17,'r':5,'t':17,'b':17}}),{'display':'none'}
    if not n_clicks:
        raise PreventUpdate
    
    
    normalizeData=3 in normalizeData
    pca = PCA(random_state=42)
    bounds=dict()
    tmpdata=[]
    curNumData=0
    if selectedData:
        #make selection linear
        sel=[[],[],[],[]]
        for i in selectedData['points']:
            sel[i['curveNumber']].append(i['customdata'])
        for idx,s in enumerate(sel):
            if s:
                nsel=numpy.array(s)
                #aggregate the selection
                curData=myNormData(inputFile["/metadata/"+str(contamination_val)+"/"+str(concentration_val)+"/"+str(coral_val)+"/"+timepoints[idx]+"/hyperspectralimage"][()][nsel[:,1],nsel[:,0]],normalizeData).reshape(-1,447)
                tmpdata.append(curData)
                bounds[idx]=[curNumData,curNumData+curData.shape[0]]
                curNumData+=curData.shape[0]
            else:
                bounds[idx]=None
    else:
        for idx,timepoint in enumerate(timepoints):
            curData=myNormData(inputFile["/metadata/"+str(contamination_val)+"/"+str(concentration_val)+"/"+str(coral_val)+"/"+timepoint+"/hyperspectralimage"][()],normalizeData).reshape(-1,447)
            tmpdata.append(curData)
            bounds[idx]=[curNumData,curNumData+curData.shape[0]]
            curNumData+=curData.shape[0]
    data=numpy.vstack(tmpdata)
    ss=StandardScaler()
    data=ss.fit_transform(data)
    data=pca.fit_transform(data)
    graphs=[]
    for idx,t in enumerate(timepoints):
        if not bounds[idx]:
            continue
        graphs.append(dict(
                type="scattergl",
                y=-1*data[bounds[idx][0]:bounds[idx][1],1],
                x=-1*data[bounds[idx][0]:bounds[idx][1],0],
                mode="markers",
                name=t,
                marker={
                    "color":colors[idx],
                }
                ))
    coeffs = pca.components_.T[:,:2]/pca.components_.T[:,:2].max()*data.max()
    annotations=[]
    for i in range(coeffs.shape[0]):
        annotations.append(dict(
            x=-1*coeffs[i,0],  # arrows' head
            y=-1*coeffs[i,1],  # arrows' head
            ax=0,  # arrows' tail
            ay=0,  # arrows' tail
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            text='',
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=nm2Color(i+383)#'rgb'+str(tuple((numpy.array(matplotlib.cm.Spectral(i)[:3])*255).astype(int)))
        ))
    
    return dict(data=graphs,layout={'plot_bgcolor':'rgb(40,40,40)','margin':{'l':17,'r':5,'t':17,'b':17},'annotations': annotations}),{'display':'block'}

