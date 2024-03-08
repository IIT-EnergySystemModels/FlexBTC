# Developed by

#    Juan F. Gutierrez-Guerra
#    Instituto de Investigacion Tecnologica
#    Escuela Tecnica Superior de Ingenieria - ICAI
#    UNIVERSIDAD PONTIFICIA COMILLAS
#    Alberto Aguilera 23
#    28015 Madrid, Spain
#    juan.gutierrez@iit.comillas.edu

#%% Libraries
import datetime
import os
import math
import time          # count clock time
import psutil        # access the number of CPUs
import pandas           as pd
import pyomo.environ    as pyo
from   pyomo.environ    import Set, RangeSet, Param, Var, Binary, UnitInterval, NonNegativeIntegers, PositiveIntegers, NonNegativeReals, Reals, Any, Constraint, ConcreteModel, Objective, maximize, Suffix
from   pyomo.opt        import SolverFactory
from   pyomo.dataportal import DataPortal
from   collections      import defaultdict


for i in range(0, 117):
    print('-', end="")

print('\nProgram for Optimizing the Operation of BTC Plant - Version 1.7.2 - March 7, 2024')
print('#### Non-commercial use only ####')

for i in range(0, 117):
    print('-', end="")

StartTime = time.time()

DirName    = os.path.dirname(__file__)
CaseName   = ('CEMEXH2_Alicante')
SolverName = 'gurobi'

_path = os.path.join(DirName, CaseName)

#%% Model declaration
mBTC = ConcreteModel('Program for Optimizing the Operation of BTC Plant - Version 1.7.2 - March 7, 2024')

#%% Reading the sets
dictSets = DataPortal()
dictSets.load(filename=_path+'/BTC_Dict_Period_'     +CaseName+'.csv', set='p'   , format='set')
dictSets.load(filename=_path+'/BTC_Dict_LoadLevel_'  +CaseName+'.csv', set='t'   , format='set')
dictSets.load(filename=_path+'/BTC_Dict_Generation_' +CaseName+'.csv', set='g'   , format='set')
dictSets.load(filename=_path+'/BTC_Dict_StartUp_'    +CaseName+'.csv', set='l'   , format='set')

mBTC.pp = Set(initialize=dictSets['p'], ordered=True,  doc='periods'     )
mBTC.tt = Set(initialize=dictSets['t'], ordered=True,  doc='load levels' )
mBTC.gg = Set(initialize=dictSets['g'], ordered=False, doc='units'       )
mBTC.ll = Set(initialize=dictSets['l'], ordered=True,  doc='su types'    )

#%% Reading data from CSV files
dfParameter          = pd.read_csv(_path+'/BTC_Data_Parameter_'         +CaseName+'.csv', index_col=[0    ])
dfDuration           = pd.read_csv(_path+'/BTC_Data_Duration_'          +CaseName+'.csv', index_col=[0    ])
dfPeriod             = pd.read_csv(_path+'/BTC_Data_Period_'            +CaseName+'.csv', index_col=[0    ])
dfDemand             = pd.read_csv(_path+'/BTC_Data_Demand_'            +CaseName+'.csv', index_col=[0,1  ])
dfEnergyCost         = pd.read_csv(_path+'/BTC_Data_EnergyCost_'        +CaseName+'.csv', index_col=[0,1  ])
dfEnergyPrice        = pd.read_csv(_path+'/BTC_Data_EnergyPrice_'       +CaseName+'.csv', index_col=[0,1  ])
dfFuelCost           = pd.read_csv(_path+'/BTC_Data_FuelCost_'          +CaseName+'.csv', index_col=[0    ])
dfGeneration         = pd.read_csv(_path+'/BTC_Data_Generation_'        +CaseName+'.csv', index_col=[0    ])
dfStartUp            = pd.read_csv(_path+'/BTC_Data_StartUp_'           +CaseName+'.csv', index_col=[0,1  ])
dfInitialCond        = pd.read_csv(_path+'/BTC_Data_InitialConditions_' +CaseName+'.csv', index_col=[0    ])
dfSDTraject          = pd.read_csv(_path+'/BTC_Data_SDTrajectory_'      +CaseName+'.csv', index_col=[0,1  ])
dfSUTraject          = pd.read_csv(_path+'/BTC_Data_SUTrajectories_'    +CaseName+'.csv', index_col=[0,1,2])
dfORPrice            = pd.read_csv(_path+'/BTC_Data_ORPrice_'           +CaseName+'.csv', index_col=[0,1  ])

# substitute NaN by 0
dfParameter.fillna   (0.0, inplace=True)
dfDuration.fillna    (0.0, inplace=True)
dfPeriod.fillna      (0.0, inplace=True)
dfDemand.fillna      (0.0, inplace=True)
dfEnergyCost.fillna  (0.0, inplace=True)
dfEnergyPrice.fillna (0.0, inplace=True)
dfFuelCost.fillna    (0.0, inplace=True)
dfGeneration.fillna  (0.0, inplace=True)
dfStartUp.fillna     (0.0, inplace=True)
dfInitialCond.fillna (0.0, inplace=True)
dfORPrice.fillna     (0.0, inplace=True)

#%% general parameters
pPNSCost             = dfParameter  ['PNSCost'             ].iloc[0]                # cost of electric energy not served      [EUR/MWh]
pQNSCost             = dfParameter  ['QNSCost'             ].iloc[0]                # cost of heat not served                 [EUR/MWh]
pCO2Cost             = dfParameter  ['CO2Cost'             ].iloc[0]                # cost of CO2 emission                    [EUR/tonCO2]
pGCPCapacity         = dfParameter  ['GCPCapacity'         ].iloc[0]                # grid connection point capacity          [MW]
pHPOperation         = dfParameter  ['HPOperation'         ].iloc[0]                # heat pump connected to BTC indicator    [Yes/No]
pTimeStep            = dfParameter  ['TimeStep'            ].iloc[0].astype('int')  # duration of the unit time step          [h]
pAnnualDiscRate      = dfParameter  ['AnnualDiscountRate'  ].iloc[0]                # annual discount rate                    [p.u.]
pUpReserveActivation = dfParameter  ['UpReserveActivation' ].iloc[0]                # upward   reserve activation             [p.u.]
pDwReserveActivation = dfParameter  ['DwReserveActivation' ].iloc[0]                # downward reserve activation             [p.u.]
pMinRatioDwUp        = dfParameter  ['MinRatioDwUp'        ].iloc[0]                # min ratio down up operating reserves    [p.u.]
pMaxRatioDwUp        = dfParameter  ['MaxRatioDwUp'        ].iloc[0]                # max ratio down up operating reserves    [p.u.]
pEconomicBaseYear    = dfParameter  ['EconomicBaseYear'    ].iloc[0]                # economic base year                      [year]
pDuration            = dfDuration   ['Duration'            ] * pTimeStep            # duration of load levels                 [h]
pPeriodWeight        = dfPeriod     ['Weight'              ].astype('int')          # weights of periods                      [p.u.]
pPDemand             = dfDemand     ['PDemand'             ]                        # electric power demand                   [MW]
pQDemand             = dfDemand     ['QDemand'             ]                        # heat demand                             [MW]
pEnergyCost          = dfEnergyCost ['Cost'                ]                        # energy cost                             [EUR/MWh]
pEnergyPrice         = dfEnergyPrice['Price'               ]                        # energy price                            [EUR/MWh]
pBiomassCost         = dfFuelCost   ['Biomass'             ]                        # biomass cost                            [EUR/MWh]
pGasCost             = dfFuelCost   ['NaturalGas'          ]                        # natural gas cost                        [EUR/MWh]
pHydrogenCost        = dfFuelCost   ['Hydrogen'            ]                        # hydrogen cost                           [EUR/MWh]
pIndCogeneration     = dfGeneration ['IndCogeneration'     ]                        # generator is a cogeneration unit        [Yes/No]
pIndHeatPump         = dfGeneration ['IndHeatPump'         ]                        # generator is a heat pump unit           [Yes/No]
pIndFuel             = dfGeneration ['IndFuel'             ]                        # fuel type                               ['Biomass'/'NaturalGas'/'Hydrogen']
pIndOperReserve      = dfGeneration ['IndOperReserve'      ]                        # contribution to operating reserve       [Yes/No]
pMaxPower            = dfGeneration ['MaximumPower'        ]                        # rated maximum power                     [MW]
pMinPower            = dfGeneration ['MinimumPower'        ]                        # rated minimum power                     [MW]
pMaxQ                = dfGeneration ['MaxHeatOutput'       ]                        # maximum heat output                     [MW]
pMinQ                = dfGeneration ['MinHeatOutput'       ]                        # minimum heat output                     [MW]
pEfficiency          = dfGeneration ['Efficiency'          ]                        # efficiency                              [p.u.]
pCOP                 = dfGeneration ['COP'                 ]                        # heat pump COP                           [p.u.]
pPQSlope             = dfGeneration ['Pqslope'             ]                        # slope of the linear P-Q curve           [MW/MW]
pPQYIntercept        = dfGeneration ['Pqyintercept'        ]                        # y-intercept of the linear P-Q curve     [MW]
pRampUp              = dfGeneration ['RampUp'              ]                        # ramp up   rate                          [MW/h]
pRampDw              = dfGeneration ['RampDown'            ]                        # ramp down rate                          [MW/h]
pUpTime              = dfGeneration ['UpTime'              ]                        # minimum up   time                       [h]
pDwTime              = dfGeneration ['DownTime'            ]                        # minimum down time                       [h]
pShutDownCost        = dfGeneration ['ShutDownCost'        ]                        # shutdown cost                           [EUR]
pSDDuration          = dfGeneration ['SDDuration'          ]                        # duration of the shut-down ramp process  [h]
pCO2ERate            = dfGeneration ['CO2EmissionRate'     ]                        # emission  rate                          [tCO2/MWh]
pPowerSyn            = dfStartUp    ['PowerSyn'            ]                        # P at which the unit is synchronized     [MW]
pStartUpCost         = dfStartUp    ['StartUpCost'         ]                        # startup  cost                           [EUR]
pSUDuration          = dfStartUp    ['SUDuration'          ]                        # duration of the start-up l ramp process [h]
pOffDuration         = dfStartUp    ['OffDuration'         ]                        # min periods before beginning of SU l    [h]
pCommitment0         = dfInitialCond['Commitment0'         ]                        # initial commitment state of the unit    {0.1}
pUpTime0             = dfInitialCond['UpTime0'             ]                        # nr of hours that has been up before     [h]
pDwTime0             = dfInitialCond['DownTime0'           ]                        # nr of hours that has been dw before     [h]
pOutput0             = dfInitialCond['p0'                  ]                        # initial power output                    [MW]
pPsdi                = dfSDTraject  ['Psd_i'               ]                        # P at beginning of ith interval of SD    [MW]
pPsui                = dfSUTraject  ['Psu_i'               ]                        # P at beginning of ith interval of SU    [MW]
pORPrice             = dfORPrice    ['Price'               ]                        # secondary reserve price                 [EUR/MW]

# compute the Demand as the mean over the time step load levels and assign it to active load levels.
pPDemand             = pPDemand.rolling      (pTimeStep).mean()
pQDemand             = pQDemand.rolling      (pTimeStep).mean()
pEnergyCost          = pEnergyCost.rolling   (pTimeStep).mean()
pEnergyPrice         = pEnergyPrice.rolling  (pTimeStep).mean()
pORPrice             = pORPrice.rolling      (pTimeStep).mean()

pPDemand.fillna      (0.0, inplace=True)
pQDemand.fillna      (0.0, inplace=True)
pEnergyCost.fillna   (0.0, inplace=True)
pEnergyPrice.fillna  (0.0, inplace=True)
pORPrice.fillna      (0.0, inplace=True)
pPsdi.fillna         (0.0, inplace=True)
pPsui.fillna         (0.0, inplace=True)

if pTimeStep > 1:
    # assign duration 0 to load levels not being considered, active load levels are at the end of every pTimeStep
    for i in range(pTimeStep-2,-1,-1):
        pDuration.iloc[[range(i,len(mBTC.tt),pTimeStep)]] = 0

ReadingDataTime = time.time() - StartTime
StartTime       = time.time()
print('\nReading input data....................... ', round(ReadingDataTime), 's')

# replacing string values by numerical values
idxDict        = dict()
idxDict[0    ] = 0
idxDict[0.0  ] = 0
idxDict['No' ] = 0
idxDict['NO' ] = 0
idxDict['no' ] = 0
idxDict['N'  ] = 0
idxDict['n'  ] = 0
idxDict['Yes'] = 1
idxDict['YES'] = 1
idxDict['yes'] = 1
idxDict['Y'  ] = 1
idxDict['y'  ] = 1

if pHPOperation in idxDict:
    pHPOperation = idxDict[pHPOperation]
pIndCogeneration = pIndCogeneration.map(idxDict)
pIndHeatPump     = pIndHeatPump    .map(idxDict)
pIndOperReserve  = pIndOperReserve .map(idxDict)


# defining subsets
mBTC.p    = Set(initialize=mBTC.pp,         ordered=True , doc='periods'           , filter=lambda mBTC,pp: pp in mBTC.pp and pPeriodWeight[pp] >  0.0                                                     )
mBTC.t    = Set(initialize=mBTC.tt,         ordered=True , doc='load levels'       , filter=lambda mBTC,tt: tt in mBTC.tt and pDuration    [tt] >  0                                                       )
mBTC.g    = Set(initialize=mBTC.gg,         ordered=False, doc='generating units'  , filter=lambda mBTC,gg: gg in mBTC.gg and pMaxPower    [gg] >  0.0 and pIndHeatPump    [gg] < 1                        )
mBTC.gc   = Set(initialize=mBTC.gg,         ordered=False, doc='cogeneration units', filter=lambda mBTC,gg: gg in mBTC.gg and pMaxPower    [gg] >  0.0 and pIndCogeneration[gg] > 0                        )
mBTC.gx   = Set(initialize=mBTC.gg,         ordered=False, doc='thermal units'     , filter=lambda mBTC,gg: gg in mBTC.gg and pMaxPower    [gg] >  0.0 and pIndCogeneration[gg] < 1 and pCO2ERate[gg] > 0.0)
mBTC.gh   = Set(initialize=mBTC.gg,         ordered=False, doc='heat pump units'   , filter=lambda mBTC,gg: gg in mBTC.gg and pMaxPower    [gg] >  0.0 and pIndHeatPump    [gg] > 0                        )
mBTC.l    = Set(initialize=mBTC.ll,         ordered=True , doc='su types'          , filter=lambda mBTC,ll: ll in mBTC.ll                                                                                  )
mBTC.l2   = Set(initialize=dfStartUp.index, ordered=True , doc='su types index'                                                                                                                            )
mBTC.l2c  = Set(initialize=lambda m: list(mBTC.l2 - {(g,l) for g,l in mBTC.gx*mBTC.l}), ordered=True, doc='BTC su types index'                                                                             )
mBTC.lx   = Set(initialize=lambda m: list(mBTC.l  - {l     for g,l in mBTC.l2c      }), ordered=True, doc='thermal units su types'                                                                         )
mBTC.lc   = Set(initialize=lambda m: list(mBTC.l  - mBTC.lx                          ), ordered=True, doc='BTC su types'                                                                                   )
mBTC.sdi  = Set(initialize=lambda m: list(range(1, int(pSDDuration.max()) + 2)       ), ordered=True, doc='shut-down time periods'                                                                         )
mBTC.sui  = Set(initialize=lambda m: list(range(1, pSUDuration.max()      + 2)       ), ordered=True, doc='start-up  time periods'                                                                         )

g2l = defaultdict(list)
for g,l in mBTC.l2:
    g2l[l].append(g)

# instrumental sets
mBTC.pg     = [(p, g            ) for p, g             in mBTC.p    * mBTC.g          ]
mBTC.pt     = [(p, t            ) for p, t             in mBTC.p    * mBTC.t          ]
mBTC.ptg    = [(p, t, g         ) for p, t, g          in mBTC.pt   * mBTC.g          ]
mBTC.ptgg   = [(p, t, gg        ) for p, t, gg         in mBTC.pt   * mBTC.gg         ]
mBTC.ptgc   = [(p, t, gc        ) for p, t, gc         in mBTC.pt   * mBTC.gc         ]
mBTC.ptgclc = [(p, t, gc, lc    ) for p, t, gc, lc     in mBTC.ptgc * mBTC.lc         ]
mBTC.ptgx   = [(p, t, gx        ) for p, t, gx         in mBTC.pt   * mBTC.gx         ]
mBTC.ptgxlx = [(p, t, gx, lx    ) for p, t, gx, lx     in mBTC.ptgx * mBTC.lx         ]
mBTC.ptgh   = [(p, t, gh        ) for p, t, gh         in mBTC.pt   * mBTC.gh         ]
mBTC.gl     = [(      g,  l     ) for       g , l      in mBTC.l2                     ]
mBTC.pgl    = [(p,    g,  l     ) for p,    g , l      in mBTC.p    * mBTC.gl         ]
mBTC.ptgl   = [(p, t, g,  l     ) for p, t, g , l      in mBTC.p    * mBTC.t * mBTC.gl]
mBTC.gsdi   = [(      g,     sdi) for       g ,    sdi in mBTC.g    * mBTC.sdi        ]
mBTC.glsui  = [(      g,  l, sui) for       g , l, sui in mBTC.gl   * mBTC.sui        ]

# getting the current year
pCurrentYear = datetime.date.today().year

if pAnnualDiscRate == 0.0:
    pDiscountFactor = pd.Series(data=[                        pPeriodWeight[p]                                                                                          for p in mBTC.p], index=mBTC.p)
else:
    pDiscountFactor = pd.Series(data=[((1.0+pAnnualDiscRate)**pPeriodWeight[p]-1.0) / (pAnnualDiscRate*(1.0+pAnnualDiscRate)**(pPeriodWeight[p]-1+p-pEconomicBaseYear)) for p in mBTC.p], index=mBTC.p)

# minimum up- and downtime and maximum shift time converted to an integer number of time steps
pUpTime = round(pUpTime / pTimeStep).astype('int')
pDwTime = round(pDwTime / pTimeStep).astype('int')

# drop levels with duration 0
pDuration      = pDuration.loc     [mBTC.t      ]
pPDemand       = pPDemand.loc      [mBTC.pt     ]
pQDemand       = pQDemand.loc      [mBTC.pt     ]
pEnergyCost    = pEnergyCost.loc   [mBTC.pt     ]
pEnergyPrice   = pEnergyPrice.loc  [mBTC.pt     ]
pORPrice       = pORPrice.loc      [mBTC.pt     ]

# drop 0 values
pPsui            = pPsui.loc           [pPsui != 0.0]
pPQSlope         = pPQSlope.loc        [mBTC.gc     ]
pPQYIntercept    = pPQYIntercept.loc   [mBTC.gc     ]
pIndOperReserve  = pIndOperReserve.loc [mBTC.gc     ]
pCOP             = pCOP.loc            [mBTC.gh     ]

# drop parameters that do not apply to heat pump units
pIndFuel       = pIndFuel.loc      [mBTC.g      ]
pEfficiency    = pEfficiency.loc   [mBTC.g      ]
pRampUp        = pRampUp.loc       [mBTC.g      ]
pRampDw        = pRampDw.loc       [mBTC.g      ]
pUpTime        = pUpTime.loc       [mBTC.g      ]
pDwTime        = pDwTime.loc       [mBTC.g      ]
pShutDownCost  = pShutDownCost.loc [mBTC.g      ]
pCO2ERate      = pCO2ERate.loc     [mBTC.g      ]
pPowerSyn      = pPowerSyn.loc     [mBTC.g      ]
pStartUpCost   = pStartUpCost.loc  [mBTC.g      ]
pSDDuration    = pSDDuration.loc   [mBTC.g      ]
pSUDuration    = pSUDuration.loc   [mBTC.g      ]
pOffDuration   = pOffDuration.loc  [mBTC.g      ]
pCommitment0   = pCommitment0.loc  [mBTC.g      ]
pUpTime0       = pUpTime0.loc      [mBTC.g      ]
pDwTime0       = pDwTime0.loc      [mBTC.g      ]
pOutput0       = pOutput0.loc      [mBTC.g      ]
pPsdi          = pPsdi.loc         [mBTC.g      ]
pPsui          = pPsui.loc         [mBTC.g      ]

# this option avoids a warning in the following assignments
pd.options.mode.chained_assignment = None

## Auxiliary parameters

# used for initial conditions [h]
pUpTimeR      = pd.Series(data=[max(0.0,(pUpTime[g] - pUpTime0[g]) *      pCommitment0[g])  for g in mBTC.g], index=mBTC.g)
pDwTimeR      = pd.Series(data=[max(0.0,(pDwTime[g] - pDwTime0[g]) * (1 - pCommitment0[g])) for g in mBTC.g], index=mBTC.g)

# maximum start up duration per generator type [h]
for gc, lc in mBTC.gc*mBTC.lc:
    pMaxSUDuration_gc = pSUDuration[gc,lc].max()

for gx, lx in mBTC.gx*mBTC.lx:
    pMaxSUDuration_gx = pSUDuration[gx,lx].max()

# assigns fuel cost [EUR/MWh]
pFuelCost = pd.Series(index=mBTC.g, dtype='float64')
for g in mBTC.g:
    if pIndFuel  [g] == 'Biomass':
        pFuelCost[g] =  float(pBiomassCost.iloc[0])
    elif pIndFuel[g] == 'Hydrogen':
        pFuelCost[g] =  float(pHydrogenCost.iloc[0])
    else:
        pFuelCost[g] =  float(pGasCost.iloc[0])

# Grid connection capacity [MW]
if pGCPCapacity == 0.0:
    pGCPCapacity = math.inf

# maximum power 2nd block [MW]
for g in mBTC.g:
    pMaxPower2ndBlock = pd.Series(data=[(pMaxPower[gg] - pMinPower[gg]) for gg in mBTC.gg], index=mBTC.gg)

# max and min achievable power output in gc units (above min output) [MW]
for gc in mBTC.gc:
    pMaxP         = pd.Series(data=[(pMaxPower[gc] - pMaxQ[gc]) for gc in mBTC.gc], index=mBTC.gc)
    pMinP         = pd.Series(data=[(pMinPower[gc] - pMinQ[gc]) for gc in mBTC.gc], index=mBTC.gc)
    pMaxP2ndBlock = pd.Series(data=[(pMaxP    [gc] - pMinP[gc]) for gc in mBTC.gc], index=mBTC.gc)
    pMaxQ2ndBlock = pd.Series(data=[(pMaxQ    [gc] - pMinQ[gc]) for gc in mBTC.gc], index=mBTC.gc)

# max and min heat and power inputs in heat pump units [MW]
for gh in mBTC.gh:
    pMaxQInHP     = pd.Series(data=[(pMaxPower[gh] * (1 - (1 / pCOP[gh]))) for gh in mBTC.gh], index=mBTC.gh)
    pMinQInHP     = pd.Series(data=[(pMinPower[gh] * (1 - (1 / pCOP[gh]))) for gh in mBTC.gh], index=mBTC.gh)
    pMaxPInHP     = pd.Series(data=[(pMaxPower[gh] - pMaxQInHP[gh])        for gh in mBTC.gh], index=mBTC.gh)
    pMinPInHP     = pd.Series(data=[(pMinPower[gh] - pMinQInHP[gh])        for gh in mBTC.gh], index=mBTC.gh)


## Parameters

#General
mBTC.pPDemand             = Param(mBTC.pt    , initialize=pPDemand.to_dict()            , within=Reals,               doc='Power demand'                          )
mBTC.pQDemand             = Param(mBTC.pt    , initialize=pQDemand.to_dict()            , within=Reals,               doc='Heat  demand'                          )
mBTC.pPeriodWeight        = Param(mBTC.p     , initialize=pPeriodWeight.to_dict()       , within=NonNegativeIntegers, doc='Period weight'   , mutable=True        )
mBTC.pDiscountFactor      = Param(mBTC.p     , initialize=pDiscountFactor.to_dict()     , within=NonNegativeReals,    doc='Discount factor'                       )
mBTC.pDuration            = Param(mBTC.t     , initialize=pDuration.to_dict()           , within=PositiveIntegers,    doc='Duration'        , mutable=True        )
mBTC.pEnergyCost          = Param(mBTC.pt    , initialize=pEnergyCost.to_dict()         , within=NonNegativeReals,    doc='Energy cost'                           )
mBTC.pEnergyPrice         = Param(mBTC.pt    , initialize=pEnergyPrice.to_dict()        , within=NonNegativeReals,    doc='Energy price'                          )
mBTC.pFuelCost            = Param(mBTC.g     , initialize=pFuelCost.to_dict()           , within=NonNegativeReals,    doc='Fuel cost'                             )
mBTC.pORPrice             = Param(mBTC.pt    , initialize=pORPrice.to_dict()            , within=NonNegativeReals,    doc='OR price'                              )

#Parameters
mBTC.pPNSCost             = Param(             initialize=pPNSCost                      , within=NonNegativeReals,    doc='PNS cost'                              )
mBTC.pQNSCost             = Param(             initialize=pQNSCost                      , within=NonNegativeReals,    doc='QNS cost'                              )
mBTC.pCO2Cost             = Param(             initialize=pCO2Cost                      , within=NonNegativeReals,    doc='CO2 emission cost'                     )
mBTC.pGCPCapacity         = Param(             initialize=pGCPCapacity                  , within=NonNegativeReals,    doc='Grid connection point capacity'        )
mBTC.pHPOperation         = Param(             initialize=pHPOperation                  , within=UnitInterval,        doc='Ind. of heat pump connected to BTC'    )
mBTC.pTimeStep            = Param(             initialize=pTimeStep                     , within=PositiveIntegers,    doc='Unitary time step'                     )
mBTC.pAnnualDiscRate      = Param(             initialize=pAnnualDiscRate               , within=UnitInterval,        doc='Annual discount rate'                  )
mBTC.pUpReserveActivation = Param(             initialize=pUpReserveActivation          , within=UnitInterval,        doc='Proportion of up   reserve activation' )
mBTC.pDwReserveActivation = Param(             initialize=pDwReserveActivation          , within=UnitInterval,        doc='Proportion of down reserve activation' )
mBTC.pMinRatioDwUp        = Param(             initialize=pMinRatioDwUp                 , within=UnitInterval,        doc='Min ratio between up and dow  reserve' )
mBTC.pMaxRatioDwUp        = Param(             initialize=pMaxRatioDwUp                 , within=UnitInterval,        doc='Max ratio between up and down reserve' )
mBTC.pEconomicBaseYear    = Param(             initialize=pEconomicBaseYear             , within=PositiveIntegers,    doc='Base year'                             )
mBTC.pCommitment0         = Param(mBTC.g     , initialize=pCommitment0.to_dict()        , within=UnitInterval,        doc='Initial commitment'                    )
mBTC.pUpTime0             = Param(mBTC.g     , initialize=pUpTime0.to_dict()            , within=NonNegativeIntegers, doc='Initial Up   time'                     )
mBTC.pDwTime0             = Param(mBTC.g     , initialize=pDwTime0.to_dict()            , within=NonNegativeIntegers, doc='Initial Down time'                     )
mBTC.pOutput0             = Param(mBTC.g     , initialize=pOutput0.to_dict()            , within=NonNegativeReals,    doc='Initial power output'                  )
mBTC.pUpTimeR             = Param(mBTC.g     , initialize=pUpTimeR.to_dict()            , within=NonNegativeIntegers, doc='Up   time R'                           )
mBTC.pDwTimeR             = Param(mBTC.g     , initialize=pDwTimeR.to_dict()            , within=NonNegativeIntegers, doc='Down time R'                           )

#Generation
mBTC.pIndCogeneration     = Param(mBTC.gg    , initialize=pIndCogeneration.to_dict()    , within=UnitInterval,        doc='Indicator of cogeneration unit'        )
mBTC.pIndHeatPump         = Param(mBTC.gg    , initialize=pIndHeatPump.to_dict()        , within=UnitInterval,        doc='Indicator of heat pump    unit'        )
mBTC.pIndOperReserve      = Param(mBTC.gc    , initialize=pIndOperReserve.to_dict()     , within=UnitInterval,        doc='Indicator of operating reserve'        )
mBTC.pMaxPower            = Param(mBTC.gg    , initialize=pMaxPower.to_dict()           , within=NonNegativeReals,    doc='Rated maximum power'                   )
mBTC.pMinPower            = Param(mBTC.gg    , initialize=pMinPower.to_dict()           , within=NonNegativeReals,    doc='Rated minimum power'                   )
mBTC.pMaxPower2ndBlock    = Param(mBTC.gg    , initialize=pMaxPower2ndBlock.to_dict()   , within=NonNegativeReals,    doc='Second block  power'                   )
mBTC.pMaxP2ndBlock        = Param(mBTC.gc    , initialize=pMaxP2ndBlock.to_dict()       , within=NonNegativeReals,    doc='CHP second block electric power'       )
mBTC.pMaxQ2ndBlock        = Param(mBTC.gc    , initialize=pMaxQ2ndBlock.to_dict()       , within=NonNegativeReals,    doc='CHP second block thermal  power'       )
mBTC.pEfficiency          = Param(mBTC.g     , initialize=pEfficiency.to_dict()         , within=UnitInterval,        doc='Round-trip efficiency'                 )
mBTC.pMaxQ                = Param(mBTC.gg    , initialize=pMaxQ.to_dict()               , within=NonNegativeReals,    doc='Maximum achievable      heat output'   )
mBTC.pMinQ                = Param(mBTC.gg    , initialize=pMinQ.to_dict()               , within=NonNegativeReals,    doc='Minimum achievable      heat output'   )
mBTC.pMaxP                = Param(mBTC.gc    , initialize=pMaxP.to_dict()               , within=NonNegativeReals,    doc='Maximum achievable BTC power output'   )
mBTC.pMinP                = Param(mBTC.gc    , initialize=pMinP.to_dict()               , within=NonNegativeReals,    doc='Minimum achievable BTC power output'   )
mBTC.pCOP                 = Param(mBTC.gh    , initialize=pCOP.to_dict()                , within=NonNegativeIntegers, doc='Heat pump COP'                         )
mBTC.pPQSlope             = Param(mBTC.gc    , initialize=pPQSlope.to_dict()            , within=Reals,               doc='Slope of linear PQ curve'              )
mBTC.pPQYIntercept        = Param(mBTC.gc    , initialize=pPQYIntercept .to_dict()      , within=Reals,               doc='Y-Intercept of linear PQ curve'        )
mBTC.pRampUp              = Param(mBTC.g     , initialize=pRampUp.to_dict()             , within=NonNegativeReals,    doc='Ramp up   rate'                        )
mBTC.pRampDw              = Param(mBTC.g     , initialize=pRampDw.to_dict()             , within=NonNegativeReals,    doc='Ramp down rate'                        )
mBTC.pUpTime              = Param(mBTC.g     , initialize=pUpTime.to_dict()             , within=NonNegativeIntegers, doc='Up   time'                             )
mBTC.pDwTime              = Param(mBTC.g     , initialize=pDwTime.to_dict()             , within=NonNegativeIntegers, doc='Down time'                             )
mBTC.pShutDownCost        = Param(mBTC.g     , initialize=pShutDownCost.to_dict()       , within=NonNegativeReals,    doc='Shutdown cost'                         )
mBTC.pSDDuration          = Param(mBTC.g     , initialize=pSDDuration.to_dict()         , within=NonNegativeIntegers, doc='Duration of SD l ramp process'         )
mBTC.pCO2ERate            = Param(mBTC.g     , initialize=pCO2ERate.to_dict()           , within=NonNegativeReals,    doc='Emission Rate'                         )

#StartUp
mBTC.pPowerSyn            = Param(mBTC.gl    , initialize=pPowerSyn.to_dict()           , within=NonNegativeReals,    doc='P at which the unit is synchronized'   )
mBTC.pStartUpCost         = Param(mBTC.gl    , initialize=pStartUpCost.to_dict()        , within=NonNegativeReals,    doc='Startup  cost'                         )
mBTC.pSUDuration          = Param(mBTC.gl    , initialize=pSUDuration.to_dict()         , within=NonNegativeIntegers, doc='Duration of SU l ramp process'         )
mBTC.pOffDuration         = Param(mBTC.gl    , initialize=pOffDuration.to_dict()        , within=NonNegativeIntegers, doc='Min periods before beginning of SU l'  )
mBTC.pPsui                = Param(mBTC.glsui , initialize=pPsui.to_dict()               , within=NonNegativeReals,    doc='P at beginning of ith interval of SU'  )

#ShutDown
mBTC.pPsdi                = Param(mBTC.gsdi  , initialize=pPsdi.to_dict()               , within=NonNegativeReals,    doc='P at beginning of ith interval of SD'  )


## Variables

mBTC.vOutput              = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g :(0.0, mBTC.pMaxPower2ndBlock[g ]), doc='output at the end of t, above min output      [MW]')
mBTC.vEnergy              = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g :(0.0, mBTC.pMaxPower        [g ]), doc='energy production at the end of t             [MWh]')
mBTC.vTotalOutput         = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g :(0.0, mBTC.pMaxPower        [g ]), doc='total output      at the end of t             [MW]')
mBTC.vReserveUp           = Var(mBTC.ptgc, within=NonNegativeReals, bounds=lambda mBTC,p,t,gc:(0.0, mBTC.pMaxPower2ndBlock[gc]), doc='upward   operating reserve                    [MW]')
mBTC.vReserveDown         = Var(mBTC.ptgc, within=NonNegativeReals, bounds=lambda mBTC,p,t,gc:(0.0, mBTC.pMaxPower2ndBlock[gc]), doc='downward operating reserve                    [MW]')
mBTC.vP                   = Var(mBTC.ptgc, within=NonNegativeReals, bounds=lambda mBTC,p,t,gc:(0.0, mBTC.pMaxP2ndBlock    [gc]), doc='CHP power output above min                    [MW]')
mBTC.vQ                   = Var(mBTC.ptgc, within=NonNegativeReals, bounds=lambda mBTC,p,t,gc:(0.0, mBTC.pMaxQ2ndBlock    [gc]), doc='CHP heat  output above min                    [MW]')
mBTC.vPOutput             = Var(mBTC.ptgc, within=NonNegativeReals, bounds=lambda mBTC,p,t,gc:(0.0, mBTC.pMaxP            [gc]), doc='power output      at the end of t             [MW]')
mBTC.vQOutput             = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g :(0.0, mBTC.pMaxQ            [g ]), doc='heat  output      at the end of t             [MW]')
mBTC.vPFS                 = Var(mBTC.ptgc, within=           Reals                                                             , doc='BTC power output final schedule (after HP)    [MW]')
mBTC.vQFS                 = Var(mBTC.ptgg, within=NonNegativeReals, bounds=lambda mBTC,p,t,gg:(0.0, mBTC.pMaxQ            [gg]), doc='heat      output final schedule               [MW]')
mBTC.vQHP                 = Var(mBTC.ptgh, within=NonNegativeReals, bounds=lambda mBTC,p,t,gh:(0.0, mBTC.pMaxPower2ndBlock[gh]), doc='heat output heat pump above min               [MW]')
mBTC.vQInHP               = Var(mBTC.ptgh, within=NonNegativeReals, bounds=lambda mBTC,p,t,gh:(0.0,      pMaxQInHP        [gh]), doc='heat input  heat pump                         [MW]')
mBTC.vQOutHP              = Var(mBTC.ptgh, within=NonNegativeReals, bounds=lambda mBTC,p,t,gh:(0.0, mBTC.pMaxQ            [gh]), doc='heat output heat pump                         [MW]')
mBTC.vPInHP               = Var(mBTC.ptgh, within=NonNegativeReals, bounds=lambda mBTC,p,t,gh:(0.0,      pMaxPInHP        [gh]), doc='power input heat pump                         [MW]')
mBTC.vCommitment          = Var(mBTC.ptg , within=Binary          , initialize=                0                               , doc='commitment of the unit during t (1 if up)    {0,1}')
mBTC.vStartUp             = Var(mBTC.ptg , within=UnitInterval    , initialize=                0.0                             , doc='start-up   of the unit (1 if it starts in t) [0,1]')
mBTC.vShutDown            = Var(mBTC.ptg , within=UnitInterval    , initialize=                0.0                             , doc='shut-down  of the unit (1 if is shuts  in t) [0,1]')
mBTC.vSUType              = Var(mBTC.ptgl, within=UnitInterval    , initialize=                0.0                             , doc='start-up type l in t   (1 if it starts in t) [0,1]')
mBTC.vOnlineStates        = Var(mBTC.ptg , within=UnitInterval    , initialize=                0.0                             , doc='online states of unit g                      [0,1]')
mBTC.vPowerBuy            = Var(mBTC.pt  , within=NonNegativeReals, bounds=lambda mBTC,p,t   :(0.0, mBTC.pGCPCapacity         ), doc='power buy                                     [MW]')
mBTC.vPowerSell           = Var(mBTC.pt  , within=NonNegativeReals, bounds=lambda mBTC,p,t   :(0.0, mBTC.pGCPCapacity         ), doc='power sell                                    [MW]')
mBTC.vQExcess             = Var(mBTC.ptgg, within=NonNegativeReals, bounds=lambda mBTC,p,t,gg:(0.0, mBTC.pMaxQ            [gg]), doc='excess produced heat                          [MW]')
mBTC.vFuel                = Var(mBTC.ptg , within=NonNegativeReals                                                             , doc='fuel consumption flow                        [MWh]')
mBTC.vIncome              = Var(mBTC.pt  , within=NonNegativeReals                                                             , doc='hourly system income                         [EUR]')
mBTC.vCost                = Var(mBTC.pt  , within=NonNegativeReals                                                             , doc='hourly system cost                           [EUR]')
mBTC.vTotalRevenue        = Var(           within=           Reals                                                             , doc='total system revenue                         [EUR]')
mBTC.vTotalCost           = Var(           within=NonNegativeReals                                                             , doc='total system cost                            [EUR]')
mBTC.vTotalIncome         = Var(           within=NonNegativeReals                                                             , doc='total system income                          [EUR]')

nFixedVariables = 0.0

for p,t,gc in mBTC.ptgc:
    if mBTC.pIndOperReserve[gc] != 1:
        mBTC.vReserveUp    [p,t,gc].fix(0.0)
        mBTC.vReserveDown  [p,t,gc].fix(0.0)
        nFixedVariables += 2

for p,t,gh in mBTC.ptgh:
    if mBTC.pHPOperation != 1:
        mBTC.vQHP          [p,t,gh].fix(0.0)
        mBTC.vQInHP        [p,t,gh].fix(0.0)
        mBTC.vQOutHP       [p,t,gh].fix(0.0)
        mBTC.vPInHP        [p,t,gh].fix(0.0)
        mBTC.vQFS          [p,t,gh].fix(0.0)
        mBTC.vQExcess      [p,t,gh].fix(0.0)
        nFixedVariables += 6

for p,t,gh in mBTC.ptgh:
    if mBTC.pHPOperation == 1:
        mBTC.vQFS          [p,t,gc].fix(0.0)
        nFixedVariables += 1

mBTC.nFixedVariables = Param(initialize=round(nFixedVariables), within=NonNegativeIntegers, doc='Number of fixed variables')

for p,t,gc in mBTC.ptgc:
    if mBTC.pHPOperation != 1:
        mBTC.vPFS[p,t,gc].setlb(0.0)
        [mBTC.vPFS[p,t,gc].setub(mBTC.pMaxP[gc]) for gc in mBTC.gc]


SettingUpDataTime = time.time() - StartTime
StartTime         = time.time()
print('Setting up input data.................... ', round(SettingUpDataTime), 's')


#%% Mathematical Formulation

#Start-up type

def eStartUpType1(mBTC,p,t,gc,lc):
    if lc < mBTC.lc.last() and mBTC.t.ord(t) >= pOffDuration[gc,mBTC.lc.next(lc)]:
        return mBTC.vSUType[p,t,gc,lc] <= sum(mBTC.vShutDown[p,(t-i),gc] for i in range(pOffDuration[gc,lc],(pOffDuration[gc,mBTC.lc.next(lc)]-1)+1))
    else:
        return Constraint.Skip
mBTC.eStartUpType1 = Constraint(mBTC.ptgclc      , rule=eStartUpType1,      doc='start-up type (1a)')

def eStartUpType2(mBTC,p,t,gx,lx):
    if lx < mBTC.lx.last() and mBTC.t.ord(t) >= pOffDuration[gx,mBTC.lx.next(lx)] and len(mBTC.lx) > 1:
        return mBTC.vSUType[p,t,gx,lx] <= sum(mBTC.vShutDown[p,(t-i),gx] for i in range(pOffDuration[gx,lx],(pOffDuration[gx,mBTC.lx.next(lx)]-1)+1))
    else:
        return Constraint.Skip
mBTC.eStartUpType2 = Constraint(mBTC.ptgxlx      , rule=eStartUpType2,      doc='start-up type (1b)')


#Only one SU type is selected when the unit starts up

def eUnitStartUp1(mBTC,p,t,gc):
    return sum(mBTC.vSUType[p,t,gc,lc] for lc in mBTC.lc) == mBTC.vStartUp[p,t,gc]
mBTC.eUnitStartUp1 = Constraint(mBTC.ptgc        , rule=eUnitStartUp1,      doc='only one SU type is selected (2a)')

def eUnitStartUp2(mBTC,p,t,gx):
    return sum(mBTC.vSUType[p,t,gx,lx] for lx in mBTC.lx) == mBTC.vStartUp[p,t,gx]
mBTC.eUnitStartUp2 = Constraint(mBTC.ptgx        , rule=eUnitStartUp2,      doc='only one SU type is selected (2b)')


#Minimum Up Time

def eMinUpTime(mBTC,p,t,g):
    if mBTC.t.ord(t) >= pUpTime[g]:
        return sum(mBTC.vStartUp [p,t,g] for t in list(mBTC.t)[(mBTC.t.ord(t) - pUpTime[g] + 1): (mBTC.t.ord(t))]) <=     mBTC.vCommitment[p,t,g]
    else:
        return Constraint.Skip
mBTC.eMinUpTime = Constraint(mBTC.ptg            , rule=eMinUpTime,         doc='min up time (3)')


#Minimum Down Time

def eMinDwTime(mBTC,p,t,g):
    if mBTC.t.ord(t) >= pDwTime[g]:
        return sum(mBTC.vShutDown[p,t,g] for t in list(mBTC.t)[(mBTC.t.ord(t) - pDwTime[g] + 1): (mBTC.t.ord(t))]) <= 1 - mBTC.vCommitment[p,t,g]
    else:
        return Constraint.Skip
mBTC.eMinDwTime = Constraint(mBTC.ptg            , rule=eMinDwTime,         doc='min down time (4)')


#Commitment, SU and SD

def eCommitment(mBTC,p,t,g):
    if t > mBTC.t.first():
        return mBTC.vCommitment[p,t,g] - mBTC.vCommitment[p,mBTC.t.prev(t),g] == mBTC.vStartUp[p,t,g] - mBTC.vShutDown[p,t,g]
    else:
        return Constraint.Skip
mBTC.eCommitment = Constraint(mBTC.ptg           , rule=eCommitment,        doc='commitment, SU and SD (5)')


#Capacity Limits

def eMaxCapacityLimit(mBTC,p,t,g):
    if t < mBTC.t.last():
        return ((mBTC.vOutput[p,t,g] + mBTC.vReserveUp  [p,t,gc]) / mBTC.pMaxPower2ndBlock[g]) <= (mBTC.vCommitment[p,t,g] - mBTC.vShutDown[p,mBTC.t.next(t),g])
    else:
        return Constraint.Skip
mBTC.eMaxCapacityLimit = Constraint(mBTC.ptg     , rule=eMaxCapacityLimit,  doc='max capacity limit (6a)')

def eMinCapacityLimit(mBTC,p,t,g):
    return ((mBTC.vOutput[p,t,g] - mBTC.vReserveDown[p,t,gc]) / mBTC.pMaxPower2ndBlock[g]) >= 0.0
mBTC.eMinCapacityLimit = Constraint(mBTC.ptg     , rule=eMinCapacityLimit,  doc='min capacity limit (6b)')


#Operating Ramp Constraints

def eOperatingRampUp(mBTC,p,t,g):
    if t > mBTC.t.first():
        return ((mBTC.vOutput[p,t,g] + mBTC.vReserveUp  [p,t,gc] - mBTC.vOutput[p,mBTC.t.prev(t),g] - mBTC.vReserveDown[p,mBTC.t.prev(t),gc]) / mBTC.pDuration[t]) <=  mBTC.pRampUp[g]
    else:
        return Constraint.Skip
mBTC.eOperatingRampUp = Constraint(mBTC.ptg      , rule=eOperatingRampUp,   doc='operating ramp up (7a)')

def eOperatingRampDw(mBTC,p,t,g):
    if t > mBTC.t.first():
        return ((mBTC.vOutput[p,t,g] - mBTC.vReserveDown[p,t,gc] - mBTC.vOutput[p,mBTC.t.prev(t),g] + mBTC.vReserveUp  [p,mBTC.t.prev(t),gc]) / mBTC.pDuration[t]) >= -mBTC.pRampDw[g]
    else:
        return Constraint.Skip
mBTC.eOperatingRampDw = Constraint(mBTC.ptg      , rule=eOperatingRampDw,   doc='operating ramp dw (7b)')


#PowerSchedule

def eTotalPOutputCHP(mBTC,p,t,gc):
    if t < (mBTC.t.last() - pMaxSUDuration_gc):
        return mBTC.vTotalOutput[p,t,gc] == ((mBTC.pMinPower[gc] * (mBTC.vCommitment[p,t,gc] + mBTC.vStartUp[p,mBTC.t.next(t),gc])) + mBTC.vOutput [p,t,gc]  +
                                             + mBTC.vReserveUp[p,t,gc] - mBTC.vReserveDown[p,t,gc]                                                           +
                                             sum((mBTC.pPsdi[gc,i] * mBTC.vShutDown[p,(t-i+2),gc]) for i in range(2, int(pSDDuration[gc])+1))                +
                                             sum(sum((mBTC.pPsui[gc,lc,i] * mBTC.vSUType[p,(t-i+pSUDuration[gc,lc]+2),gc,lc]) for i in range(1,pSUDuration[gc,lc]+1)) for gc,lc in mBTC.gc*mBTC.lc))
    elif   (mBTC.t.last() - pMaxSUDuration_gc) <= t <= mBTC.t.last():
        return mBTC.vTotalOutput[p,t,gc] == ((mBTC.pMinPower[gc]  * mBTC.vCommitment[p,t,gc]) + mBTC.vOutput [p,t,gc]  +
                                             + mBTC.vReserveUp[p,t,gc] - mBTC.vReserveDown[p,t,gc]                     +
                                             sum((mBTC.pPsdi[gc,i] * mBTC.vShutDown[p,(t-i+2),gc]) for i in range(2, int(pSDDuration[gc])+1)))
    else:
        return Constraint.Skip
mBTC.eTotalPOutputCHP = Constraint(mBTC.ptgc     , rule=eTotalPOutputCHP,   doc='CHP total power output')

def eTotalPOutputTU(mBTC,p,t,gx):
    if t < (mBTC.t.last() - pMaxSUDuration_gx):
        return mBTC.vTotalOutput[p,t,gx] == ((mBTC.pMinPower[gx] * (mBTC.vCommitment[p,t,gx] + mBTC.vStartUp[p,mBTC.t.next(t),gx])) + mBTC.vOutput[p,t,gx] +
                                            sum((mBTC.pPsdi[gx,i] * mBTC.vShutDown[p,(t-i+2),gx]) for i in range(2, int(pSDDuration[gx])+1)) +
                                            sum(sum((mBTC.pPsui[gx,lx,i] * mBTC.vSUType[p,(t-i+pSUDuration[gx,lx]+2),gx,lx]) for i in range(1,pSUDuration[gx,lx]+1)) for gx,lx in mBTC.gx*mBTC.lx))
    elif   (mBTC.t.last() - pMaxSUDuration_gx) <= t <= mBTC.t.last():
        return mBTC.vTotalOutput[p,t,gx] == ((mBTC.pMinPower[gx]  * mBTC.vCommitment[p,t,gx])                                       + mBTC.vOutput[p,t,gx] +
                                            sum((mBTC.pPsdi[gx,i] * mBTC.vShutDown[p,(t-i+2),gx]) for i in range(2, int(pSDDuration[gx])+1)))
    else:
        return Constraint.Skip
mBTC.eTotalPOutputTU = Constraint(mBTC.ptgx      , rule=eTotalPOutputTU,    doc='thermal units total power output')

def eOnlineStatesCHP(mBTC,p,t,gc):
    if mBTC.t.first() < t < (mBTC.t.last() - pMaxSUDuration_gc):
        return mBTC.vOnlineStates[p,t,gc] == (mBTC.vCommitment[p,t,gc] +
                                            sum(mBTC.vShutDown[p,(t-i+1),gc] for i in range(1, int(pSDDuration[gc])+1)) +
                                            sum(sum(mBTC.vSUType[p,(t-j+pSUDuration[gc,lc]+1),gc,lc] for j in range(1, int(pSUDuration[gc,lc])+1)) for gc,lc in mBTC.gc*mBTC.lc))
    elif (mBTC.t.last() - pMaxSUDuration_gc) <= t <= mBTC.t.last():
        return mBTC.vOnlineStates[p,t,gc] == (mBTC.vCommitment[p,t,gc] +
                                            sum(mBTC.vShutDown[p,(t-i+1),gc] for i in range(1, int(pSDDuration[gc])+1)))
    else:
        return Constraint.Skip
mBTC.eOnlineStatesCHP = Constraint(mBTC.ptgc     , rule=eOnlineStatesCHP,   doc='CHP online states')

def eOnlineStatesTU(mBTC,p,t,gx):
    if mBTC.t.first() < t < (mBTC.t.last() - pMaxSUDuration_gx):
        return mBTC.vOnlineStates[p,t,gx] == (mBTC.vCommitment[p,t,gx] +
                                            sum(mBTC.vShutDown[p,(t-i+1),gx] for i in range(1, int(pSDDuration[gx])+1)) +
                                            sum(sum(mBTC.vSUType[p,(t-j+pSUDuration[gx,lx]+1),gx,lx] for j in range(1, int(pSUDuration[gx,lx])+1)) for gx,lx in mBTC.gx*mBTC.lx))
    elif (mBTC.t.last() - pMaxSUDuration_gx) <= t <= mBTC.t.last():
        return mBTC.vOnlineStates[p,t,gx] == (mBTC.vCommitment[p,t,gx] +
                                            sum(mBTC.vShutDown[p,(t-i+1),gx] for i in range(1, int(pSDDuration[gx])+1)))
    else:
        return Constraint.Skip
mBTC.eOnlineStatesTU = Constraint(mBTC.ptgx      , rule=eOnlineStatesTU,    doc='thermal units online states')

def eEnergyProduction(mBTC,p,t,g):
    return mBTC.vEnergy[p,t,g] == mBTC.vTotalOutput[p,t,g] * mBTC.pDuration[t]
mBTC.eEnergyProduction = Constraint(mBTC.ptg     , rule=eEnergyProduction,  doc='energy production (8)')


#OperatingReserves

def eReserveMinRatioDwUp(mBTC,p,t,gc):
    return mBTC.vReserveDown[p,t,gc] >= mBTC.vReserveUp[p,t,gc] * mBTC.pMinRatioDwUp
mBTC.eReserveMinRatioDwUp = Constraint(mBTC.ptgc , rule=eReserveMinRatioDwUp, doc='minimum ratio down to up operating reserve [MW]')

def eReserveMaxRatioDwUp(mBTC,p,t,gc):
    return mBTC.vReserveDown[p,t,gc] <= mBTC.vReserveUp[p,t,gc] * mBTC.pMaxRatioDwUp
mBTC.eReserveMaxRatioDwUp = Constraint(mBTC.ptgc , rule=eReserveMaxRatioDwUp, doc='maximum ratio down to up operating reserve [MW]')


#PowerBalance

def ePOutput(mBTC,p,t,gc):
    return mBTC.vPOutput[p,t,gc] == mBTC.vP[p,t,gc] + (mBTC.pMinP[gc] * mBTC.vCommitment[p,t,gc])
mBTC.ePOutput = Constraint(mBTC.ptgc             , rule=ePOutput,           doc='CHP electric power balance [MW]')

def eQOutput(mBTC,p,t,gc):
    return mBTC.vQOutput[p,t,gc] == mBTC.vQ[p,t,gc] + (mBTC.pMinQ[gc] * mBTC.vCommitment[p,t,gc])
mBTC.eQOutput = Constraint(mBTC.ptgc             , rule=eQOutput,           doc='CHP thermal power balance [MW]')

def ePQCurve(mBTC,p,t,gc):
    return mBTC.vP[p,t,gc] == mBTC.pPQSlope[gc] * mBTC.vQ[p,t,gc]
mBTC.ePQCurve = Constraint(mBTC.ptgc             , rule=ePQCurve,           doc='BTC PQ relation [MW]')

def eBTCBalance1(mBTC,p,t,gc):
    return mBTC.vTotalOutput[p,t,gc] == mBTC.vPOutput[p,t,gc] + mBTC.vQOutput[p,t,gc] + mBTC.vReserveUp[p,t,gc] - mBTC.vReserveDown[p,t,gc]
mBTC.eBTCBalance1 = Constraint(mBTC.ptgc         , rule=eBTCBalance1,       doc='BTC output balance 1 [MW]')

def eBTCBalance2(mBTC,p,t,gc):
    if mBTC.pHPOperation == 1:
        return mBTC.vPOutput[p,t,gc] == mBTC.vPFS[p,t,gc] + mBTC.vPInHP[p,t,gh]
    else:
        return mBTC.vPOutput[p,t,gc] == mBTC.vPFS[p,t,gc]
mBTC.eBTCBalance2 = Constraint(mBTC.ptgc         , rule=eBTCBalance2,       doc='BTC output balance 2 [MW]')

def eBTCBalance3(mBTC,p,t,gc):
    if mBTC.pHPOperation == 1:
        return mBTC.vQOutput[p,t,gc] == mBTC.vQInHP[p,t,gh] + mBTC.vQExcess[p,t,gc]
    else:
        return mBTC.vQOutput[p,t,gc] == mBTC.vQFS  [p,t,gc] + mBTC.vQExcess[p,t,gc]
mBTC.eBTCBalance3 = Constraint(mBTC.ptgc         , rule=eBTCBalance3,       doc='BTC output balance 3 [MW]')

def eHPBalance(mBTC,p,t,gh):
    if mBTC.pHPOperation == 1:
        return mBTC.vQOutHP[p,t,gh] == (pMinPower[gh] * mBTC.vCommitment[p,t,gc]) + mBTC.vQHP[p,t,gh]
    else:
        return Constraint.Skip
mBTC.eHPBalance = Constraint(mBTC.ptgh           , rule=eHPBalance,         doc='heat pump balance [MW]')
def eHPBalance1(mBTC,p,t,gh):
    if mBTC.pHPOperation == 1:
        return mBTC.vQInHP[p,t,gh] == mBTC.vQOutHP[p,t,gh] * (1 - (1 / mBTC.pCOP[gh]))
    else:
        return Constraint.Skip
mBTC.eHPBalance1 = Constraint(mBTC.ptgh          , rule=eHPBalance1,        doc='heat pump balance 1 [MW]')

def eHPBalance2(mBTC,p,t,gh):
    if mBTC.pHPOperation == 1:
        return mBTC.vPInHP[p,t,gh] == mBTC.vQOutHP[p,t,gh] - mBTC.vQInHP[p,t,gh]
    else:
        return Constraint.Skip
mBTC.eHPBalance2 = Constraint(mBTC.ptgh          , rule=eHPBalance2,        doc='heat pump balance 2 [MW]')

def eHPBalance3(mBTC,p,t,gh):
    if mBTC.pHPOperation == 1:
        return mBTC.vQOutHP[p,t,gh] == mBTC.vQFS[p,t,gh] + mBTC.vQExcess[p,t,gh]
    else:
        return Constraint.Skip
mBTC.eHPBalance3 = Constraint(mBTC.ptgh          , rule=eHPBalance3,        doc='heat pump balance 3 [MW]')

def eXBalance(mBTC,p,t,gx):
    return mBTC.vTotalOutput[p,t,gx] == mBTC.vQFS[p,t,gx] + mBTC.vQExcess[p,t,gx]
mBTC.eXBalance = Constraint(mBTC.ptgx            , rule=eXBalance,          doc='thermal units output balance [MW]')

def ePBalance(mBTC,p,t):
    return (sum(mBTC.vPFS[p,t,gc] for gc in mBTC.gc) + mBTC.vPowerBuy[p,t] - mBTC.vPowerSell[p,t]) == mBTC.pPDemand[p,t]
mBTC.ePBalance = Constraint(mBTC.pt              , rule=ePBalance,          doc='electric power balance [MW]')

def eGridConnection(mBTC,p,t):
    return mBTC.vPowerBuy[p,t] + mBTC.vPowerSell[p,t] <= mBTC.pGCPCapacity
mBTC.eGridConnection = Constraint(mBTC.pt        , rule=eGridConnection,    doc='grid connection limit [MW]')

def eQBalance(mBTC,p,t):
    return (sum(mBTC.vQFS[p,t,gg] for gg in mBTC.gg)) == mBTC.pQDemand[p,t]
mBTC.eQBalance = Constraint(mBTC.pt              , rule=eQBalance,          doc='thermal power balance [MW]')

def eFuelFlow(mBTC,p,t,g):
    return mBTC.vFuel[p,t,g] == (mBTC.vEnergy[p,t,g]) / mBTC.pEfficiency[g]
mBTC.eFuelCost = Constraint(mBTC.ptg,              rule=eFuelFlow,          doc='fuel flow')


#Initial Conditions

def eIniOutput(mBTC,p,t,g):
    if t == mBTC.t.first():
        return mBTC.vTotalOutput[p,t,g] == mBTC.pOutput0[g]
    else:
        return Constraint.Skip
mBTC.eIniOutput = Constraint(mBTC.ptg            , rule=eIniOutput,         doc='initial power output')

def eIniOState(mBTC,p,t,g):
    if t == mBTC.t.first():
        return mBTC.vOnlineStates[p,t,g] == mBTC.pCommitment0[g]
    else:
        return Constraint.Skip
mBTC.eIniOState = Constraint(mBTC.ptg            , rule=eIniOState,         doc='initial online state')

def eInitialCom(mBTC,p,t,g):
    if (mBTC.pUpTimeR[g] + mBTC.pDwTimeR[g]) >= 1.0:
        if mBTC.t.ord(t) <= (mBTC.pUpTimeR[g] + mBTC.pDwTimeR[g]):
            return mBTC.vCommitment[p,t,g] == mBTC.pCommitment0[g]
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
mBTC.eInitialCom = Constraint(mBTC.ptg           , rule=eInitialCom,        doc='initial commitment (14)')

def eIniSUType1(mBTC,p,t,gc,lc):
    if mBTC.pDwTime0[gc] >= 2.0:
        if lc < mBTC.lc.last() and ((pOffDuration[gc, mBTC.lc.next(lc)] - mBTC.pDwTime0[gc]) < t < pOffDuration[gc, mBTC.lc.next(lc)]):
            return mBTC.vSUType[p,t,gc,lc] == 0.0
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
mBTC.eIniSUType1 = Constraint(mBTC.ptgclc        , rule=eIniSUType1,        doc='initial su type (15a)')

def eIniSUType2(mBTC,p,t,gx,lx):
    if len(mBTC.lx) > 1 and mBTC.pDwTime0[gx] >= 2.0:
        if lx < mBTC.lx.last() and ((pOffDuration[gx, mBTC.lx.next(lx)] - mBTC.pDwTime0[gx]) < t < pOffDuration[gx, mBTC.lx.next(lx)]):
            return mBTC.vSUType[p,t,gx,lx] == 0.0
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
mBTC.eIniSUType2 = Constraint(mBTC.ptgxlx        , rule=eIniSUType2,        doc='initial su type (15b)')

def eIniSUType3(mBTC,p,t,gc,lc):
    if t <= pSUDuration[gc,lc]:
        return mBTC.vSUType[p,t,gc,lc] == 0.0
    else:
        return Constraint.Skip
mBTC.eIniSUType3 = Constraint(mBTC.ptgclc        , rule=eIniSUType3,        doc='initial su type (15c)')

def eIniSUType4(mBTC,p,t,gx,lx):
    if t <= pSUDuration[gx,lx]:
        return mBTC.vSUType[p,t,gx,lx] == 0.0
    else:
        return Constraint.Skip
mBTC.eIniSUType4 = Constraint(mBTC.ptgxlx        , rule=eIniSUType4,        doc='initial su type (15d)')


#%% Objective Function

def eIncome(mBTC,p,t):
    return mBTC.vIncome[p,t] ==     ((mBTC.pEnergyPrice[p,t] * mBTC.pDuration[t] * mBTC.vPowerSell  [p,t   ]) +
                                 sum((mBTC.pORPrice    [p,t]                     * mBTC.vReserveUp  [p,t,gc]) +
                                     (mBTC.pORPrice    [p,t]                     * mBTC.vReserveDown[p,t,gc]) for gc in mBTC.gc if mBTC.pIndOperReserve[gc] == 1))
mBTC.eIncome = Constraint(mBTC.pt                , rule=eIncome,            doc='hourly system income [EUR]')

def eCost(mBTC,p,t):
    return mBTC.vCost[p,t] == (sum(((mBTC.pFuelCost            [g    ]                      * mBTC.vFuel       [p,t,g    ])                                 +
                               sum( (mBTC.pStartUpCost         [gc,lc]                      * mBTC.vSUType     [p,t,gc,lc])  for gc,lc  in mBTC.gc*mBTC.lc) +
                               sum( (mBTC.pStartUpCost         [gx,lx]                      * mBTC.vSUType     [p,t,gx,lx])  for gx,lx  in mBTC.gx*mBTC.lx) +
                                    (mBTC.pShutDownCost        [g    ]                      * mBTC.vShutDown   [p,t,g    ])                                 +
                                  ( (mBTC.pCO2Cost * mBTC.pCO2ERate[g]) * mBTC.pDuration[t] * mBTC.vTotalOutput[p,t,g    ])) for g      in mBTC.g         ) +
                               sum( (mBTC.pQNSCost                      * mBTC.pDuration[t] * mBTC.vQExcess    [p,t,gg   ])  for gg     in mBTC.gg        ) +
                                    (mBTC.pEnergyCost          [p,t  ]  * mBTC.pDuration[t] * mBTC.vPowerBuy   [p,t      ]))
mBTC.eCost = Constraint(mBTC.pt                  , rule=eCost,              doc='hourly system cost [EUR]')

def eTotalIncome(mBTC):
    return mBTC.vTotalIncome == sum(mBTC.vIncome[p,t] for p,t in mBTC.pt)
mBTC.eTotalIncome = Constraint(                    rule=eTotalIncome,       doc='total system income [EUR]')

def eTotalCost(mBTC):
    return mBTC.vTotalCost   == sum(mBTC.vCost  [p,t] for p,t in mBTC.pt)
mBTC.eTotalCost = Constraint(                      rule=eTotalCost,         doc='total system cost [EUR]')

def eTotalRevenue(mBTC):
    return mBTC.vTotalRevenue == mBTC.vTotalIncome - mBTC.vTotalCost
mBTC.eTotalRevenue = Constraint(                   rule=eTotalRevenue,      doc='total system revenue [EUR]')


def eObjFunction(mBTC):
    return mBTC.vTotalRevenue
mBTC.eObjFunction = Objective(rule=eObjFunction  , sense=maximize,          doc='objective function [EUR]')


GeneratingOFTime = time.time() - StartTime
StartTime        = time.time()
print('Generating objective function............ ', round(GeneratingOFTime), 's')


#%% Problem solving

mBTC.write(_path+'/BTC_'+CaseName+'.lp', io_options={'symbolic_solver_labels': True})   # create lp-format file
Solver = SolverFactory(SolverName)                                                      # select solver
Solver.options['LogFile'       ] = _path+'/BTC_'+CaseName+'.log'
Solver.options['OutputFlag'    ] = 0
Solver.options['IISFile'       ] = _path+'/BTC_'+CaseName+'.ilp'                        # should be uncommented to show results of IIS
#Solver.options['Method'       ] = 2                                                      # barrier method
Solver.options['MIPGap'        ] = 0.01
Solver.options['Threads'       ] = int((psutil.cpu_count(logical=True) + psutil.cpu_count(logical=False))/2)
Solver.options['TimeLimit'     ] = 7200
Solver.options['IterationLimit'] = 7200000
SolverResults = Solver.solve(mBTC, tee=False)                                           # tee=True displays the output of the solver
#SolverResults.write()                                                                    # summary of the solver results

for p,t,g in mBTC.ptg:
    mBTC.vCommitment  [p,t,g].fix(round(mBTC.vCommitment  [p,t,g]()))
    mBTC.vStartUp     [p,t,g].fix(round(mBTC.vStartUp     [p,t,g]()))
    mBTC.vShutDown    [p,t,g].fix(round(mBTC.vShutDown    [p,t,g]()))
    mBTC.vOnlineStates[p,t,g].fix(round(mBTC.vOnlineStates[p,t,g]()))

for p,t,g,l in mBTC.ptgl:
    mBTC.vSUType    [p,t,g,l].fix(round(mBTC.vSUType    [p,t,g,l]()))

Solver.options['relax_integrality'] = 1                                                 # introduced to show results of the dual variables
mBTC.dual = Suffix(direction=Suffix.IMPORT)
SolverResults = Solver.solve(mBTC, tee=False)                                           # tee=True displays the output of the solver
# SolverResults.write()                                                                   # summary of the solver results

SolvingTime = time.time() - StartTime
StartTime   = time.time()

print('***** Period: ' + str(p) + ' ******')
print('Problem size............................. ', mBTC.model().nconstraints(), 'constraints, ', mBTC.model().nvariables() - mBTC.nFixedVariables + 1, 'variables')
print('Solution time............................ ', round(SolvingTime), 's')
print('Total system income...................... ', round(mBTC.vTotalIncome() *1e-6, ndigits=2), '[MEUR]')
print('Total system cost........................ ', round(mBTC.vTotalCost()   *1e-6, ndigits=2), '[MEUR]')
print('Total system revenue..................... ', round(mBTC.vTotalRevenue()*1e-6, ndigits=2), '[MEUR]')


#%% Final Power Schedule

OfflineStates = pd.Series(data=[1.0 - mBTC.vOnlineStates[p,t,g]() for p,t,g in mBTC.ptg], index=pd.MultiIndex.from_tuples(mBTC.ptg))


#%% Output Results

Output_Income = pd.DataFrame({'Total Income [MEUR]' : [round(mBTC.vTotalIncome() *1e-6, 2)]})
Output_Costs  = pd.DataFrame({'Total Cost [MEUR]'   : [round(mBTC.vTotalCost()   *1e-6, 2)]})
Output_Rev    = pd.DataFrame({'Total Revenue [MEUR]': [round(mBTC.vTotalRevenue()*1e-6, 2)]})
OutputResults = pd.concat([Output_Income, Output_Costs, Output_Rev], axis=1).to_csv(_path + '/BTC_Result_02_CostsSummary_' + CaseName + '.csv', sep=',', index=False)

OfflineStates.to_frame(   name='p.u.').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='p.u.').rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Offline {x}'           ).to_csv(_path+'/BTC_Result_OfflineStates_'+CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vEnergy[p,t,g]()     for p,t,g   in mBTC.ptg],            index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='MWh' ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MWh' ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Energy {x} [MWh]'      ).to_csv(_path+'/BTC_Result_Energy_'       +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vCommitment[p,t,g]() for p,t,g   in mBTC.ptg],            index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='p.u.').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='p.u.').rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Commitment {x}'        ).to_csv(_path+'/BTC_Result_Commitment_'   +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vStartUp[p,t,g]()    for p,t,g   in mBTC.ptg],            index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='p.u.').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='p.u.').rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'StartUp {x}'           ).to_csv(_path+'/BTC_Result_StartUp_'      +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vShutDown[p,t,g]()   for p,t,g   in mBTC.ptg],            index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='p.u.').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='p.u.').rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ShutDown {x}'          ).to_csv(_path+'/BTC_Result_ShutDown_'     +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vSUType[p,t,g,l]()   for p,t,g,l in mBTC.ptgl],           index=pd.MultiIndex.from_tuples(mBTC.ptgl))
OutputResults.to_frame(   name='p.u.').reset_index().pivot_table( index=['level_0','level_1'], columns='level_3', values='p.u.').rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'SUType {x}'            ).to_csv(_path+'/BTC_Result_SUType_'       +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vTotalOutput[p,t,g]() for p,t,g  in mBTC.ptg],            index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'TotalOutput {x} [MW]'  ).to_csv(_path+'/BTC_Result_TotalOutput_'  +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vOnlineStates[p,t,g]() for p,t,g in mBTC.ptg],            index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='p.u.').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='p.u.').rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Online {x}'            ).to_csv(_path+'/BTC_Result_OnlineStates_' +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vPowerBuy[p,t]()       for p,t   in mBTC.pt],             index=pd.MultiIndex.from_tuples(mBTC.pt))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'],                    values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'PowerBuy {x}'          ).to_csv(_path+'/BTC_Result_PowerBuy_'     +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vPowerSell[p,t]()      for p,t   in mBTC.pt],             index=pd.MultiIndex.from_tuples(mBTC.pt))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1']                   , values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'PowerSell {x}'         ).to_csv(_path+'/BTC_Result_PowerSell_'    +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vPOutput[p,t,gc]()     for p,t,gc in mBTC.ptgc],          index=pd.MultiIndex.from_tuples(mBTC.ptgc))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'POutput {x} [MW]'      ).to_csv(_path+'/BTC_Result_PowerOutput_'  +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vPFS[p,t,gc]()         for p,t,gc in mBTC.ptgc],          index=pd.MultiIndex.from_tuples(mBTC.ptgc))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'PFS {x} [MW]'          ).to_csv(_path+'/BTC_Result_PowerFS_'      +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vQOutput[p,t,g]()      for p,t,g in mBTC.ptg],            index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QOutput {x} [MW]'      ).to_csv(_path+'/BTC_Result_HeatOutput_'   +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vReserveUp[p,t,gc]()   for p,t,gc in mBTC.ptgc],          index=pd.MultiIndex.from_tuples(mBTC.ptgc))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ReserveUp {x} [MW]'    ).to_csv(_path+'/BTC_Result_ReserveUp_'   +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vReserveDown[p,t,gc]() for p,t,gc in mBTC.ptgc],          index=pd.MultiIndex.from_tuples(mBTC.ptgc))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ReserveDw {x} [MW]'    ).to_csv(_path+'/BTC_Result_ReserveDw_'   +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vQFS[p,t,gg]()         for p,t,gg in mBTC.ptgg],          index=pd.MultiIndex.from_tuples(mBTC.ptgg))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QFS {x} [MW]'          ).to_csv(_path+'/BTC_Result_HeatFS_'       +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vQExcess[p,t,gg]()     for p,t,gg in mBTC.ptgg],          index=pd.MultiIndex.from_tuples(mBTC.ptgg))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QExcess {x} [MW]'      ).to_csv(_path+'/BTC_Result_HeatExcess_'   +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[(mBTC.pFuelCost[g]*mBTC.vFuel[p,t,g]()) for p,t,g in mBTC.ptg],index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='EUR' ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='EUR' ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'FuelCost {x} [EUR]'    ).to_csv(_path+'/BTC_Result_FuelCost_'     +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vCost[p,t]()           for p,t   in mBTC.pt],             index=pd.MultiIndex.from_tuples(mBTC.pt))
OutputResults.to_frame(   name='EUR'  ).reset_index().pivot_table( index=['level_0','level_1'],                   values='EUR' ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Cost {x}'              ).to_csv(_path+'/BTC_Result_Costs_'        +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vIncome[p,t]()         for p,t   in mBTC.pt],             index=pd.MultiIndex.from_tuples(mBTC.pt))
OutputResults.to_frame(   name='EUR'  ).reset_index().pivot_table( index=['level_0','level_1'],                   values='EUR' ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Income {x}'            ).to_csv(_path+'/BTC_Result_Incomes_'      +CaseName+'.csv', sep=',')

if mBTC.pHPOperation == 1:

    OutputResults = pd.Series(data=[mBTC.vQInHP[p,t,gh]()       for p,t,gh in mBTC.ptgh],          index=pd.MultiIndex.from_tuples(mBTC.ptgh))
    OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QInHP {x} [MW]'        ).to_csv(_path+'/BTC_Result_HeatInHP_'     +CaseName+'.csv', sep=',')

    OutputResults = pd.Series(data=[mBTC.vPInHP[p,t,gh]()       for p,t,gh in mBTC.ptgh],          index=pd.MultiIndex.from_tuples(mBTC.ptgh))
    OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'PInHP {x} [MW]'        ).to_csv(_path+'/BTC_Result_PowerInHP_'    +CaseName+'.csv', sep=',')

    OutputResults = pd.Series(data=[mBTC.vQOutHP[p,t,gh]()      for p,t,gh in mBTC.ptgh],          index=pd.MultiIndex.from_tuples(mBTC.ptgh))
    OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QOutHP {x} [MW]'       ).to_csv(_path+'/BTC_Result_HeatOutHP_'    +CaseName+'.csv', sep=',')


# Creating a dataframe with all output results

dfCommitment        = pd.read_csv(_path + '/BTC_Result_Commitment_'     + CaseName + '.csv', index_col=[0, 1])
dfStartUp           = pd.read_csv(_path + '/BTC_Result_Startup_'        + CaseName + '.csv', index_col=[0, 1])
dfShutDown          = pd.read_csv(_path + '/BTC_Result_ShutDown_'       + CaseName + '.csv', index_col=[0, 1])
dfEnergy            = pd.read_csv(_path + '/BTC_Result_Energy_'         + CaseName + '.csv', index_col=[0, 1])
dfTotalOutput       = pd.read_csv(_path + '/BTC_Result_TotalOutput_'    + CaseName + '.csv', index_col=[0, 1])
dfOnlineStates      = pd.read_csv(_path + '/BTC_Result_OnlineStates_'   + CaseName + '.csv', index_col=[0, 1])
dfOfflineStates     = pd.read_csv(_path + '/BTC_Result_OfflineStates_'  + CaseName + '.csv', index_col=[0, 1])
dfSUType            = pd.read_csv(_path + '/BTC_Result_SUType_'         + CaseName + '.csv', index_col=[0, 1])
dfPowerBuy          = pd.read_csv(_path + '/BTC_Result_PowerBuy_'       + CaseName + '.csv', index_col=[0, 1])
dfPowerSell         = pd.read_csv(_path + '/BTC_Result_PowerSell_'      + CaseName + '.csv', index_col=[0, 1])
dfPOutput           = pd.read_csv(_path + '/BTC_Result_PowerOutput_'    + CaseName + '.csv', index_col=[0, 1])
dfPFS               = pd.read_csv(_path + '/BTC_Result_PowerFS_'        + CaseName + '.csv', index_col=[0, 1])
dfReserveUp         = pd.read_csv(_path + '/BTC_Result_ReserveUp_'      + CaseName + '.csv', index_col=[0, 1])
dfReserveDw         = pd.read_csv(_path + '/BTC_Result_ReserveDw_'      + CaseName + '.csv', index_col=[0, 1])
dfQOutput           = pd.read_csv(_path + '/BTC_Result_HeatOutput_'     + CaseName + '.csv', index_col=[0, 1])
dfQFS               = pd.read_csv(_path + '/BTC_Result_HeatFS_'         + CaseName + '.csv', index_col=[0, 1])
dfQExcess           = pd.read_csv(_path + '/BTC_Result_HeatExcess_'     + CaseName + '.csv', index_col=[0, 1])
dfFuelCost          = pd.read_csv(_path + '/BTC_Result_FuelCost_'       + CaseName + '.csv', index_col=[0, 1])
dfCosts             = pd.read_csv(_path + '/BTC_Result_Costs_'          + CaseName + '.csv', index_col=[0, 1])
dfIncomes           = pd.read_csv(_path + '/BTC_Result_Incomes_'        + CaseName + '.csv', index_col=[0, 1])

if mBTC.pHPOperation == 1:
    dfHeatInHP      = pd.read_csv(_path + '/BTC_Result_HeatInHP_'       + CaseName + '.csv', index_col=[0, 1])
    dfPowerInHP     = pd.read_csv(_path + '/BTC_Result_PowerInHP_'      + CaseName + '.csv', index_col=[0, 1])
    dfHeatOutHP     = pd.read_csv(_path + '/BTC_Result_HeatOutHP_'      + CaseName + '.csv', index_col=[0, 1])

if mBTC.pHPOperation == 1:
    dfs = [dfOnlineStates,
           dfOfflineStates,
           dfCommitment,
           dfStartUp,
           dfShutDown,
           dfSUType,
           dfEnergy,
           dfFuelCost,
           dfTotalOutput,
           dfPowerBuy,
           dfPowerSell,
           dfPOutput,
           dfPowerInHP,
           dfPFS,
           dfDemand['PDemand'],
           dfQOutput,
           dfHeatInHP,
           dfHeatOutHP,
           dfQExcess,
           dfQFS,
           dfDemand['QDemand'],
           dfCosts,
           dfIncomes]
else:
    dfs = [dfOnlineStates,
           dfOfflineStates,
           dfCommitment,
           dfStartUp,
           dfShutDown,
           dfSUType,
           dfEnergy,
           dfFuelCost,
           dfTotalOutput,
           dfPowerBuy,
           dfPowerSell,
           dfPOutput,
           dfPFS,
           dfReserveUp,
           dfReserveDw,
           dfDemand['PDemand'],
           dfQOutput,
           dfQFS,
           dfQExcess,
           dfDemand['QDemand'],
           dfCosts,
           dfIncomes]

rounded_dfs = []

for df in dfs:
    if isinstance(df, pd.Series):
        # Handle rounding for Series
        rounded_values = [round(value, 2) if pd.notna(value) else None for value in df.values]
        rounded_df = pd.Series(data=rounded_values, index=df.index, name=df.name)
    elif isinstance(df, pd.DataFrame):
        # Handle rounding for DataFrame
        rounded_df = df.round(2)
    else:
        # If the data type is not recognized, keep the original DataFrame
        rounded_df = df

    rounded_dfs.append(rounded_df)

dfResult = pd.concat(rounded_dfs, join='outer', axis=1).to_csv(_path + '/BTC_Result_01_Summary_' + CaseName + '.csv', sep=',', header=True, index=True)


WritingResultsTime = time.time() - StartTime
StartTime          = time.time()
print('Writing output results................... ', round(WritingResultsTime), 's')
print('Total time............................... ', round(ReadingDataTime + SettingUpDataTime + SolvingTime + WritingResultsTime), 's')
print('\n #### Non-commercial use only #### \n')
