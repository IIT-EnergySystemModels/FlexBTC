# Developed by

#    Juan F. Gutierrez-Guerra
#    Instituto de Investigacion Tecnologica
#    Escuela Tecnica Superior de Ingenieria - ICAI
#    UNIVERSIDAD PONTIFICIA COMILLAS
#    Alberto Aguilera 23
#    28015 Madrid, Spain
#    jgutierrez@comillas.edu

#%% Libraries
import datetime
import os
import math
import time             # count clock time
import psutil           # access the number of CPUs
import pandas           as pd
import pyomo.environ    as pyo
from   pyomo.environ    import Set, RangeSet, Param, Var, Binary, UnitInterval, NonNegativeIntegers, PositiveIntegers, NonNegativeReals, Reals, Constraint, ConcreteModel, Objective, maximize, Suffix
from   pyomo.opt        import SolverFactory
from   pyomo.dataportal import DataPortal
from   collections      import defaultdict


for i in range(0, 117):
    print('-', end="")

print('\nOptimizing the Operation of CHP units coupled with SGHPs - Version 2.1.0 - August 27, 2025')
print('#### Non-commercial use only ####')

for i in range(0, 117):
    print('-', end="")

StartTime = time.time()

DirName    = os.path.dirname(__file__)
CaseName   = ('Baseline')
SolverName = 'gurobi'

_path = os.path.join(DirName, CaseName)

#%% Model declaration
mBTC = ConcreteModel('Optimizing the Operation of CHP units coupled with SGHPs - Version 2.1.0 - August 27, 2025')

#%% Reading the sets
dictSets = DataPortal()
dictSets.load(filename=_path+'/BTC_Dict_Period_'      +CaseName+'.csv', set='p'   , format='set')
dictSets.load(filename=_path+'/BTC_Dict_LoadLevel_'   +CaseName+'.csv', set='t'   , format='set')
dictSets.load(filename=_path+'/BTC_Dict_Generation_'  +CaseName+'.csv', set='g'   , format='set')
dictSets.load(filename=_path+'/BTC_Dict_StartUp_'     +CaseName+'.csv', set='l'   , format='set')
dictSets.load(filename=_path+'/BTC_Dict_MarketPeriod_'+CaseName+'.csv', set='m'   , format='set')

mBTC.pp = Set(initialize=dictSets['p'], ordered=True,  doc='periods'       )
mBTC.tt = Set(initialize=dictSets['t'], ordered=True,  doc='load levels'   )
mBTC.gg = Set(initialize=dictSets['g'], ordered=False, doc='units'         )
mBTC.ll = Set(initialize=dictSets['l'], ordered=True,  doc='su types'      )
mBTC.mm = Set(initialize=dictSets['m'], ordered=True,  doc='market periods')

#%% Reading data from CSV files
dfParameter          = pd.read_csv(_path+'/BTC_Data_Parameter_'          +CaseName+'.csv', index_col=[0    ])
dfDuration           = pd.read_csv(_path+'/BTC_Data_Duration_'           +CaseName+'.csv', index_col=[0    ])
dfPeriod             = pd.read_csv(_path+'/BTC_Data_Period_'             +CaseName+'.csv', index_col=[0    ])
dfDemand             = pd.read_csv(_path+'/BTC_Data_Demand_'             +CaseName+'.csv', index_col=[0,1  ])
dfEnergyCost         = pd.read_csv(_path+'/BTC_Data_EnergyCost_'         +CaseName+'.csv', index_col=[0,1  ])
dfEnergyPrice        = pd.read_csv(_path+'/BTC_Data_EnergyPrice_'        +CaseName+'.csv', index_col=[0,1  ])
dfPowerCost          = pd.read_csv(_path+'/BTC_Data_ContractedPowerCost_'+CaseName+'.csv', index_col=[0,1  ])
dfFuelCost           = pd.read_csv(_path+'/BTC_Data_FuelCost_'           +CaseName+'.csv', index_col=[0,1  ])
dfGeneration         = pd.read_csv(_path+'/BTC_Data_Generation_'         +CaseName+'.csv', index_col=[0    ])
dfStartUp            = pd.read_csv(_path+'/BTC_Data_StartUp_'            +CaseName+'.csv', index_col=[0,1  ])
dfInitialCond        = pd.read_csv(_path+'/BTC_Data_InitialConditions_'  +CaseName+'.csv', index_col=[0    ])
dfSDTraject          = pd.read_csv(_path+'/BTC_Data_SDTrajectory_'       +CaseName+'.csv', index_col=[0,1  ])
dfSUTraject          = pd.read_csv(_path+'/BTC_Data_SUTrajectories_'     +CaseName+'.csv', index_col=[0,1,2])
dfORPrice            = pd.read_csv(_path+'/BTC_Data_ORPrice_'            +CaseName+'.csv', index_col=[0,1  ])
dfORActivation       = pd.read_csv(_path+'/BTC_Data_ORActivation_'       +CaseName+'.csv', index_col=[0,1  ])
dfMarketPeriods      = pd.read_csv(_path+'/BTC_Data_HourlyMarketPeriods_'+CaseName+'.csv', index_col=[0,1,2])

# substitute NaN by 0
dfParameter.fillna    (0.0, inplace=True)
dfDuration.fillna     (0.0, inplace=True)
dfPeriod.fillna       (0.0, inplace=True)
dfDemand.fillna       (0.0, inplace=True)
dfEnergyCost.fillna   (0.0, inplace=True)
dfEnergyPrice.fillna  (0.0, inplace=True)
dfPowerCost.fillna    (0.0, inplace=True)
dfFuelCost.fillna     (0.0, inplace=True)
dfGeneration.fillna   (0.0, inplace=True)
dfStartUp.fillna      (0.0, inplace=True)
dfInitialCond.fillna  (0.0, inplace=True)
dfORPrice.fillna      (0.0, inplace=True)
dfORActivation.fillna (0.0, inplace=True)
dfMarketPeriods.fillna(0.0, inplace=True)

#%% general parameters
pPNSCost             = dfParameter    ['PNSCost'             ].iloc[0]                # cost of electric energy not served      [EUR/MWh]
pCoolingCost         = dfParameter    ['CoolingCost'         ].iloc[0]                # cost of heat waste cooling              [EUR/MWh]
pCO2Cost             = dfParameter    ['CO2Cost'             ].iloc[0]                # cost of CO2 emission                    [EUR/tonCO2]
pGCPCapacity         = dfParameter    ['GCPCapacity'         ].iloc[0]                # grid connection point capacity          [MW]
pVAT                 = dfParameter    ['VAT'                 ].iloc[0]                # value added tax                         [p.u.]
pSET                 = dfParameter    ['SET'                 ].iloc[0]                # special electricity tax (Spain)         [p.u.]
pTimeStep            = dfParameter    ['TimeStep'            ].iloc[0].astype('int')  # duration of the unit time step          [h]
pAnnualDiscRate      = dfParameter    ['AnnualDiscountRate'  ].iloc[0]                # annual discount rate                    [p.u.]
pMinRatioDwUp        = dfParameter    ['MinRatioDwUp'        ].iloc[0]                # min ratio down up operating reserves    [p.u.]
pMaxRatioDwUp        = dfParameter    ['MaxRatioDwUp'        ].iloc[0]                # max ratio down up operating reserves    [p.u.]
pEconomicBaseYear    = dfParameter    ['EconomicBaseYear'    ].iloc[0]                # economic base year                      [year]
pDuration            = dfDuration     ['Duration'            ] * pTimeStep            # duration of load levels                 [h]
pPeriodWeight        = dfPeriod       ['Weight'              ].astype('int')          # weights of periods                      [p.u.]
pPDemand             = dfDemand       ['PDemand'             ]                        # electric power demand                   [MW]
pQDemand             = dfDemand       ['QDemand'             ]                        # heat demand                             [MW]
pEnergyCost          = dfEnergyCost   ['Cost'                ]                        # energy cost                             [EUR/MWh]
pEnergyPrice         = dfEnergyPrice  ['Price'               ]                        # energy price                            [EUR/MWh]
pPowerCost           = dfPowerCost    ['Coefficient'         ]                        # contracted power cost coefficients      [EUR/kW year]
pBiomassCost         = dfFuelCost     ['Biomass'             ]                        # biomass cost                            [EUR/MWh]
pGasCost             = dfFuelCost     ['NaturalGas'          ]                        # natural gas cost                        [EUR/MWh]
pSteamPrice          = dfFuelCost     ['SteamUtility'        ]                        # steam utility price                     [EUR/MWh]
pIndCogeneration     = dfGeneration   ['IndCogeneration'     ]                        # generator is a cogeneration      unit   [Yes/No]
pIndHeatPump         = dfGeneration   ['IndHeatPump'         ]                        # generator is a heat pump         unit   [Yes/No]
pIndHPUtility        = dfGeneration   ['IndHPUtility'        ]                        # generator is a heat pump utility unit   [Yes/No]
pHPOperation         = dfGeneration   ['HPOperation'         ]                        # heat pump connected to CHP indicator    [Yes/No]
pIndFuel             = dfGeneration   ['IndFuel'             ]                        # fuel type                               ['Biomass'/'NaturalGas'/'Hydrogen']
pIndOperReserve      = dfGeneration   ['IndOperReserve'      ]                        # contribution to operating reserve       [Yes/No]
pMaxPower            = dfGeneration   ['MaximumPower'        ]                        # rated maximum power                     [MW]
pMinPower            = dfGeneration   ['MinimumPower'        ]                        # rated minimum power                     [MW]
pMaxQ                = dfGeneration   ['MaxHeatOutput'       ]                        # maximum heat output                     [MW]
pMinQ                = dfGeneration   ['MinHeatOutput'       ]                        # minimum heat output                     [MW]
pEfficiency          = dfGeneration   ['Efficiency'          ]                        # efficiency                              [p.u.]
pCOP                 = dfGeneration   ['COP'                 ]                        # heat pump COP                           [p.u.]
pPQSlope             = dfGeneration   ['Pqslope'             ]                        # slope of the linear P-Q curve           [MW/MW]
pPQYIntercept        = dfGeneration   ['Pqyintercept'        ]                        # y-intercept of the linear P-Q curve     [MW]
pRampUp              = dfGeneration   ['RampUp'              ]                        # ramp up   rate                          [MW/h]
pRampDw              = dfGeneration   ['RampDown'            ]                        # ramp down rate                          [MW/h]
pUpTime              = dfGeneration   ['UpTime'              ]                        # minimum up   time                       [h]
pDwTime              = dfGeneration   ['DownTime'            ]                        # minimum down time                       [h]
pShutDownCost        = dfGeneration   ['ShutDownCost'        ]                        # shutdown cost                           [EUR]
pSDDuration          = dfGeneration   ['SDDuration'          ]                        # duration of the shut-down ramp process  [h]
pCO2ERate            = dfGeneration   ['CO2EmissionRate'     ]                        # emission  rate                          [tCO2/MWh]
pPowerSyn            = dfStartUp      ['PowerSyn'            ]                        # P at which the unit is synchronized     [MW]
pStartUpCost         = dfStartUp      ['StartUpCost'         ]                        # startup  cost                           [EUR]
pSUDuration          = dfStartUp      ['SUDuration'          ]                        # duration of the start-up l ramp process [h]
pOffDuration         = dfStartUp      ['OffDuration'         ]                        # min periods before beginning of SU l    [h]
pCommitment0         = dfInitialCond  ['Commitment0'         ]                        # initial commitment state of the unit    {0.1}
pUpTime0             = dfInitialCond  ['UpTime0'             ]                        # nr of hours that has been up before     [h]
pDwTime0             = dfInitialCond  ['DownTime0'           ]                        # nr of hours that has been dw before     [h]
pOutput0             = dfInitialCond  ['p0'                  ]                        # initial power output                    [MW]
pPsdi                = dfSDTraject    ['Psd_i'               ]                        # P at beginning of ith interval of SD    [MW]
pPsui                = dfSUTraject    ['Psu_i'               ]                        # P at beginning of ith interval of SU    [MW]
pAvaPrice            = dfORPrice      ['Availability'        ]                        # secondary reserve availability  price   [EUR/MW]
pActUpPrice          = dfORPrice      ['ActivationUp'        ]                        # up secondary reserve activation price   [EUR/MWh]
pActDwPrice          = dfORPrice      ['ActivationDw'        ]                        # dw secondary reserve activation price   [EUR/MWh]
pActUp               = dfORActivation ['ActivationUp'        ]                        # up secondary reserve activation prop.   [h]
pActDw               = dfORActivation ['ActivationDw'        ]                        # dw secondary reserve activation prop.   [h]
pMarketPeriods       = dfMarketPeriods['MarketPeriod'        ]                        # hourly market period                    [-]

# compute the Demand as the mean over the time step load levels and assign it to active load levels.
pPDemand             = pPDemand.rolling      (pTimeStep).mean()
pQDemand             = pQDemand.rolling      (pTimeStep).mean()
pEnergyCost          = pEnergyCost.rolling   (pTimeStep).mean()
pEnergyPrice         = pEnergyPrice.rolling  (pTimeStep).mean()
pBiomassCost         = pBiomassCost.rolling  (pTimeStep).mean()
pGasCost             = pGasCost.rolling      (pTimeStep).mean()
pSteamPrice          = pSteamPrice.rolling   (pTimeStep).mean()
pAvaPrice            = pAvaPrice.rolling     (pTimeStep).mean()
pActUpPrice          = pActUpPrice.rolling   (pTimeStep).mean()
pActDwPrice          = pActDwPrice.rolling   (pTimeStep).mean()
pActUp               = pActUp.rolling        (pTimeStep).mean()
pActDw               = pActDw.rolling        (pTimeStep).mean()
pMarketPeriods       = pMarketPeriods.rolling(pTimeStep).mean()

pPDemand.fillna      (0.0, inplace=True)
pQDemand.fillna      (0.0, inplace=True)
pEnergyCost.fillna   (0.0, inplace=True)
pEnergyPrice.fillna  (0.0, inplace=True)
pBiomassCost.fillna  (0.0, inplace=True)
pGasCost.fillna      (0.0, inplace=True)
pSteamPrice.fillna   (0.0, inplace=True)
pAvaPrice.fillna     (0.0, inplace=True)
pActUpPrice.fillna   (0.0, inplace=True)
pActDwPrice.fillna   (0.0, inplace=True)
pActUp.fillna        (0.0, inplace=True)
pActDw.fillna        (0.0, inplace=True)
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

pIndCogeneration = pIndCogeneration.map(idxDict)
pIndHeatPump     = pIndHeatPump    .map(idxDict)
pIndHPUtility    = pIndHPUtility   .map(idxDict)
pHPOperation     = pHPOperation    .map(idxDict)
pIndOperReserve  = pIndOperReserve .map(idxDict)



# defining subsets
mBTC.p      = Set(initialize=mBTC.pp,         ordered=True , doc='periods'           , filter=lambda mBTC,pp: pp in mBTC.pp and pPeriodWeight[pp] >  0.0                                                     )
mBTC.t      = Set(initialize=mBTC.tt,         ordered=True , doc='load levels'       , filter=lambda mBTC,tt: tt in mBTC.tt and pDuration    [tt] >  0                                                       )
mBTC.g      = Set(initialize=mBTC.gg,         ordered=False, doc='generating   units', filter=lambda mBTC,gg: gg in mBTC.gg and pMaxPower    [gg] >  0.0 and pIndHeatPump    [gg] < 1                        )
mBTC.gc     = Set(initialize=mBTC.gg,         ordered=False, doc='cogeneration units', filter=lambda mBTC,gg: gg in mBTC.gg and pMaxPower    [gg] >  0.0 and pIndCogeneration[gg] > 0                        )
mBTC.gcb    = Set(initialize=mBTC.gc,         ordered=False, doc='CHP + FGC    units', filter=lambda mBTC,gc: gc in mBTC.gc and pHPOperation [gc] >  0                                                       )
mBTC.gcs    = Set(initialize=mBTC.gc,         ordered=False, doc='CHP + HRSG   units', filter=lambda mBTC,gc: gc in mBTC.gc and pHPOperation [gc] == 0                                                       )
mBTC.gx     = Set(initialize=mBTC.gg,         ordered=False, doc='thermal      units', filter=lambda mBTC,gg: gg in mBTC.gg and pMaxPower    [gg] >  0.0 and pIndCogeneration[gg] < 1 and pCO2ERate[gg] > 0.0)
mBTC.gh     = Set(initialize=mBTC.gg,         ordered=False, doc='heat pump    units', filter=lambda mBTC,gg: gg in mBTC.gg and pMaxPower    [gg] >  0.0 and pIndHeatPump    [gg] > 0                        )
mBTC.gu     = Set(initialize=mBTC.gh,         ordered=False, doc='hp utility   units', filter=lambda mBTC,gh: gh in mBTC.gh and                              pIndHPUtility   [gh] > 0                        )
mBTC.l      = Set(initialize=mBTC.ll,         ordered=True , doc='su types'          , filter=lambda mBTC,ll: ll in mBTC.ll                                                                                  )
if CaseName == 'Baseline':
    mBTC.lz = Set(initialize=lambda m: list(           {'3_C_CHP1', '3_C_CHP2', '3_C_CHP3'}), ordered=True, doc='auxiliary l             index'                                                              )
mBTC.l2     = Set(initialize=dfStartUp.index, ordered=True , doc='su types index'                                                                                                                            )
mBTC.l2c    = Set(initialize=lambda m: list(mBTC.l2  - {(g,l) for g,l in mBTC.gx *mBTC.l  }), ordered=True, doc='CHP      units su types index'                                                              )
mBTC.l2b    = Set(initialize=lambda m: list(mBTC.l2c - {(g,l) for g,l in mBTC.gcs*mBTC.l  }), ordered=True, doc='BTC      units su types index'                                                              )
mBTC.l2s    = Set(initialize=lambda m: list(mBTC.l2c - {(g,l) for g,l in mBTC.gcb*mBTC.l  }), ordered=True, doc='CHP+HRSG units su types index'                                                              )
mBTC.l2x    = Set(initialize=lambda m: list(mBTC.l2  - {(g,l) for g,l in mBTC.gc *mBTC.l  }), ordered=True, doc='thermal  units su types index'                                                              )
mBTC.lx     = Set(initialize=lambda m: list(mBTC.l   - {l     for g,l in mBTC.l2c         }), ordered=True, doc='thermal  units su types'                                                                    )
mBTC.lc     = Set(initialize=lambda m: list(mBTC.l   - mBTC.lx                             ), ordered=True, doc='BTC      units su types'                                                                    )
mBTC.sdi    = Set(initialize=lambda m: list(range(1, int(pSDDuration.max()) + 2)           ), ordered=True, doc='shut-down time periods'                                                                     )
mBTC.sui    = Set(initialize=lambda m: list(range(1, pSUDuration.max()      + 2)           ), ordered=True, doc='start-up  time periods'                                                                     )
mBTC.m      = Set(initialize=mBTC.mm,         ordered=True , doc='market periods'    , filter=lambda mBTC,mm: mm in mBTC.mm                                                                                  )

g2l = defaultdict(list)
for g,l in mBTC.l2:
    g2l[l].append(g)

# instrumental sets
mBTC.pg     = [(p, g            ) for p, g               in mBTC.p    * mBTC.g          ]
mBTC.pt     = [(p, t            ) for p, t               in mBTC.p    * mBTC.t          ]
mBTC.ptg    = [(p, t, g         ) for p, t, g            in mBTC.pt   * mBTC.g          ]
mBTC.ptgg   = [(p, t, gg        ) for p, t, gg           in mBTC.pt   * mBTC.gg         ]
mBTC.ptgc   = [(p, t, gc        ) for p, t, gc           in mBTC.pt   * mBTC.gc         ]
mBTC.ptgblc = [(p, t, gcb, lc   ) for p, t, gcb, lc      in mBTC.pt   * mBTC.gcb * mBTC.lc if (gcb,lc) in mBTC.l2c]
mBTC.ptgslc = [(p, t, gcs, lc   ) for p, t, gcs, lc      in mBTC.pt   * mBTC.gcs * mBTC.lc if (gcs,lc) in mBTC.l2c]
mBTC.ptgclc = [(p, t, gc , lc   ) for p, t, gc , lc      in mBTC.pt   * mBTC.gc  * mBTC.lc if (gc ,lc) in mBTC.l2c]
mBTC.ptgh   = [(p, t, gh        ) for p, t, gh           in mBTC.pt   * mBTC.gh         ]
mBTC.ptgu   = [(p, t, gu        ) for p, t, gu           in mBTC.pt   * mBTC.gu         ]
mBTC.gl     = [(      g,  l     ) for       g  , l       in mBTC.l2                     ]
mBTC.pgl    = [(p,    g,  l     ) for p,    g  , l       in mBTC.p    * mBTC.gl         ]
mBTC.ptgl   = [(p, t, g,  l     ) for p, t, g  , l       in mBTC.p    * mBTC.t * mBTC.gl]
mBTC.gsdi   = [(      g,     sdi) for       g  ,     sdi in mBTC.g    * mBTC.sdi        ]
mBTC.glsui  = [(      g,  l, sui) for       g  , l , sui in mBTC.gl   * mBTC.sui        ]
mBTC.pm     = [(p,    m         ) for p,    m            in mBTC.p    * mBTC.m          ]
mBTC.ptm    = [(p, t, m         ) for p, t, m            in mBTC.pt   * mBTC.m          ]

# minimum up- and downtime and maximum shift time converted to an integer number of time steps
pUpTime = round(pUpTime / pTimeStep).astype('int')
pDwTime = round(pDwTime / pTimeStep).astype('int')

# drop levels with duration 0
pDuration       = pDuration.loc      [mBTC.t      ]
pPDemand        = pPDemand.loc       [mBTC.pt     ]
pQDemand        = pQDemand.loc       [mBTC.pt     ]
pEnergyCost     = pEnergyCost.loc    [mBTC.pt     ]
pEnergyPrice    = pEnergyPrice.loc   [mBTC.pt     ]
pBiomassCost    = pBiomassCost.loc   [mBTC.pt     ]
pGasCost        = pGasCost.loc       [mBTC.pt     ]
pSteamPrice     = pSteamPrice.loc    [mBTC.pt     ]
pAvaPrice       = pAvaPrice.loc      [mBTC.pt     ]
pActUpPrice     = pActUpPrice.loc    [mBTC.pt     ]
pActDwPrice     = pActDwPrice.loc    [mBTC.pt     ]
pActUp          = pActUp.loc         [mBTC.pt     ]
pActDw          = pActDw.loc         [mBTC.pt     ]

# drop 0 values
pPsui           = pPsui.loc          [pPsui != 0.0]
pPQSlope        = pPQSlope.loc       [mBTC.gc     ]
pPQYIntercept   = pPQYIntercept.loc  [mBTC.gc     ]
pIndOperReserve = pIndOperReserve.loc[mBTC.gc     ]
pIndHPUtility   = pIndHPUtility.loc  [mBTC.gh     ]
pCOP            = pCOP.loc           [mBTC.gh     ]

# drop parameters that do not apply to heat pump units
pIndFuel        = pIndFuel.loc       [mBTC.g      ]
pEfficiency     = pEfficiency.loc    [mBTC.g      ]
pRampUp         = pRampUp.loc        [mBTC.g      ]
pRampDw         = pRampDw.loc        [mBTC.g      ]
pUpTime         = pUpTime.loc        [mBTC.g      ]
pDwTime         = pDwTime.loc        [mBTC.g      ]
pShutDownCost   = pShutDownCost.loc  [mBTC.g      ]
pCO2ERate       = pCO2ERate.loc      [mBTC.g      ]
pPowerSyn       = pPowerSyn.loc      [mBTC.g      ]
pStartUpCost    = pStartUpCost.loc   [mBTC.g      ]
pSDDuration     = pSDDuration.loc    [mBTC.g      ]
pSUDuration     = pSUDuration.loc    [mBTC.g      ]
pOffDuration    = pOffDuration.loc   [mBTC.g      ]
pCommitment0    = pCommitment0.loc   [mBTC.g      ]
pUpTime0        = pUpTime0.loc       [mBTC.g      ]
pDwTime0        = pDwTime0.loc       [mBTC.g      ]
pOutput0        = pOutput0.loc       [mBTC.g      ]
pPsdi           = pPsdi.loc          [mBTC.g      ]
pPsui           = pPsui.loc          [mBTC.g      ]

# this option avoids a warning in the following assignments
pd.options.mode.chained_assignment = None

## Auxiliary parameters

# used for initial conditions [h]
pUpTimeR       = pd.Series(data=[max(0.0,(pUpTime[g] - pUpTime0[g]) *      pCommitment0[g])  for g in mBTC.g], index=mBTC.g)
pDwTimeR       = pd.Series(data=[max(0.0,(pDwTime[g] - pDwTime0[g]) * (1 - pCommitment0[g])) for g in mBTC.g], index=mBTC.g)

# maximum start up duration per generator type [h]
pMaxSUDuration = pd.Series(data=[max(pSUDuration[gc,lc] for lc in mBTC.lc if (gc,lc) in mBTC.l2c) for gc in mBTC.gc], index=mBTC.gc)

# assigns fuel cost [EUR/MWh]
pFuelCost = pd.Series(index=mBTC.ptg, dtype='float64')
for p,t,g in mBTC.ptg:
    if   pIndFuel[g] == 'Biomass':
        pFuelCost[p,t,g] = pBiomassCost[p,t]
    elif pIndFuel[g] == 'NaturalGas':
        pFuelCost[p,t,g] = pGasCost[p,t]

# Grid connection capacity [MW]
if pGCPCapacity == 0.0:
    pGCPCapacity = math.inf

# maximum power 2nd block [MW]
for g in mBTC.g:
    pMaxPower2ndBlock = pd.Series(data=[(pMaxPower[gg] - pMinPower[gg])    for gg in mBTC.gg], index=mBTC.gg)

# max and min achievable power output in gc units (above min output) [MW]
for gc in mBTC.gc:
    pMaxP         = pd.Series(data=[(pMaxPower[gc] - pMaxQ[gc])            for gc in mBTC.gc], index=mBTC.gc)
    pMinP         = pd.Series(data=[(pMinPower[gc] - pMinQ[gc])            for gc in mBTC.gc], index=mBTC.gc)
    pMaxP2ndBlock = pd.Series(data=[(pMaxP    [gc] - pMinP[gc])            for gc in mBTC.gc], index=mBTC.gc)
    pMaxQ2ndBlock = pd.Series(data=[(pMaxQ    [gc] - pMinQ[gc])            for gc in mBTC.gc], index=mBTC.gc)

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
mBTC.pDuration            = Param(mBTC.t     , initialize=pDuration.to_dict()           , within=PositiveIntegers,    doc='Duration'        , mutable=True        )
mBTC.pEnergyCost          = Param(mBTC.pt    , initialize=pEnergyCost.to_dict()         , within=NonNegativeReals,    doc='Energy cost'                           )
mBTC.pEnergyPrice         = Param(mBTC.pt    , initialize=pEnergyPrice.to_dict()        , within=NonNegativeReals,    doc='Energy price'                          )
mBTC.pPowerCost           = Param(mBTC.pm    , initialize=pPowerCost.to_dict()          , within=NonNegativeReals,    doc='Contracted power cost coefficients'    )
mBTC.pFuelCost            = Param(mBTC.ptg   , initialize=pFuelCost.to_dict()           , within=NonNegativeReals,    doc='Fuel cost'                             )
mBTC.pSteamPrice          = Param(mBTC.pt    , initialize=pSteamPrice.to_dict()         , within=NonNegativeReals,    doc='Steam utility price'                   )
mBTC.pAvaPrice            = Param(mBTC.pt    , initialize=pAvaPrice.to_dict()           , within=NonNegativeReals,    doc='Secondary reserve availability price'  )
mBTC.pActUpPrice          = Param(mBTC.pt    , initialize=pActUpPrice.to_dict()         , within=           Reals,    doc='Up second reserve activation   price'  )
mBTC.pActDwPrice          = Param(mBTC.pt    , initialize=pActDwPrice.to_dict()         , within=           Reals,    doc='Dw second reserve activation   price'  )
mBTC.pActUp               = Param(mBTC.pt    , initialize=pActUp.to_dict()              , within=UnitInterval,        doc='Up second reserve activation   propor.')
mBTC.pActDw               = Param(mBTC.pt    , initialize=pActDw.to_dict()              , within=UnitInterval,        doc='Dw second reserve activation   propor.')
mBTC.pMarketPeriods       = Param(mBTC.ptm   , initialize=pMarketPeriods.to_dict()      , within=UnitInterval,        doc='Hourly market periods'                 )

#Parameters
mBTC.pPNSCost             = Param(             initialize=pPNSCost                      , within=NonNegativeReals,    doc='PNS cost'                              )
mBTC.pCoolingCost         = Param(             initialize=pCoolingCost                  , within=NonNegativeReals,    doc='Waste heat cooling cost'               )
mBTC.pCO2Cost             = Param(             initialize=pCO2Cost                      , within=NonNegativeReals,    doc='CO2 emission cost'                     )
mBTC.pGCPCapacity         = Param(             initialize=pGCPCapacity                  , within=NonNegativeReals,    doc='Grid connection point capacity'        )
mBTC.pVAT                 = Param(             initialize=pVAT                          , within=UnitInterval,        doc='Value added tax'                       )
mBTC.pSET                 = Param(             initialize=pSET                          , within=UnitInterval,        doc='Special electricity tax'               )
mBTC.pTimeStep            = Param(             initialize=pTimeStep                     , within=PositiveIntegers,    doc='Unitary time step'                     )
mBTC.pAnnualDiscRate      = Param(             initialize=pAnnualDiscRate               , within=UnitInterval,        doc='Annual discount rate'                  )
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
mBTC.pIndCogeneration     = Param(mBTC.gg    , initialize=pIndCogeneration.to_dict()    , within=UnitInterval,        doc='Indicator of cogeneration      unit'   )
mBTC.pIndHeatPump         = Param(mBTC.gg    , initialize=pIndHeatPump.to_dict()        , within=UnitInterval,        doc='Indicator of heat pump         unit'   )
mBTC.pIndHPUtility        = Param(mBTC.gh    , initialize=pIndHPUtility.to_dict()       , within=UnitInterval,        doc='Indicator of heat pump utility unit'   )
mBTC.pHPOperation         = Param(mBTC.gg    , initialize=pHPOperation.to_dict()        , within=UnitInterval,        doc='HP coupled with a CHP          unit'  )
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
mBTC.pCOP                 = Param(mBTC.gh    , initialize=pCOP.to_dict()                , within=NonNegativeReals,    doc='Heat pump COP'                         )
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
mBTC.pMaxSUDuration       = Param(mBTC.gc    , initialize=pMaxSUDuration.to_dict()      , within=NonNegativeIntegers, doc='Max SU duration for cogeneration units')
mBTC.pOffDuration         = Param(mBTC.gl    , initialize=pOffDuration.to_dict()        , within=NonNegativeIntegers, doc='Min periods before beginning of SU l'  )
mBTC.pPsui                = Param(mBTC.glsui , initialize=pPsui.to_dict()               , within=NonNegativeReals,    doc='P at beginning of ith interval of SU'  )

#ShutDown
mBTC.pPsdi                = Param(mBTC.gsdi  , initialize=pPsdi.to_dict()               , within=NonNegativeReals,    doc='P at beginning of ith interval of SD'  )


## Variables

mBTC.vEnergy              = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g :(0.0, mBTC.pMaxPower        [g ]), doc='energy production at the end of t            [MWh]')
mBTC.vTotalOutput         = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g :(0.0, mBTC.pMaxPower        [g ]), doc='total output      at the end of t             [MW]')
mBTC.vReserveUp           = Var(mBTC.ptgc, within=NonNegativeReals, bounds=lambda mBTC,p,t,gc:(0.0, mBTC.pMaxP2ndBlock    [gc]), doc='upward   operating reserve availability       [MW]')
mBTC.vReserveDown         = Var(mBTC.ptgc, within=NonNegativeReals, bounds=lambda mBTC,p,t,gc:(0.0, mBTC.pMaxP2ndBlock    [gc]), doc='downward operating reserve availability       [MW]')
mBTC.vActivationUp        = Var(mBTC.ptgc, within=NonNegativeReals, bounds=lambda mBTC,p,t,gc:(0.0, mBTC.pMaxP2ndBlock    [gc]), doc='upward   operating reserve energy            [MWh]')
mBTC.vActivationDown      = Var(mBTC.ptgc, within=NonNegativeReals, bounds=lambda mBTC,p,t,gc:(0.0, mBTC.pMaxP2ndBlock    [gc]), doc='downward operating reserve energy            [MWh]')
mBTC.vActivation          = Var(mBTC.ptgc, within=Binary          , initialize=                0                               , doc='activation variable (1 for up, 0 for down)   {0,1}')
mBTC.vP                   = Var(mBTC.ptgc, within=NonNegativeReals, bounds=lambda mBTC,p,t,gc:(0.0, mBTC.pMaxP2ndBlock    [gc]), doc='CHP power output above min                    [MW]')
mBTC.vQ                   = Var(mBTC.ptgc, within=NonNegativeReals, bounds=lambda mBTC,p,t,gc:(0.0, mBTC.pMaxQ2ndBlock    [gc]), doc='CHP heat  output above min                    [MW]')
mBTC.vPOutput             = Var(mBTC.ptgc, within=NonNegativeReals, bounds=lambda mBTC,p,t,gc:(0.0, mBTC.pMaxP            [gc]), doc='power output      at the end of t             [MW]')
mBTC.vQOutput             = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g :(0.0, mBTC.pMaxQ            [g ]), doc='heat  output      at the end of t             [MW]')
mBTC.vPFS                 = Var(mBTC.ptgc, within=NonNegativeReals                                                             , doc='BTC power output final schedule (after HP)    [MW]')
mBTC.vQFS                 = Var(mBTC.ptgg, within=NonNegativeReals, bounds=lambda mBTC,p,t,gg:(0.0, mBTC.pMaxQ            [gg]), doc='heat      output final schedule               [MW]')
mBTC.vQHP                 = Var(mBTC.ptgh, within=NonNegativeReals, bounds=lambda mBTC,p,t,gh:(0.0, mBTC.pMaxPower2ndBlock[gh]), doc='heat output heat pump above min               [MW]')
mBTC.vQInHP               = Var(mBTC.ptgh, within=NonNegativeReals, bounds=lambda mBTC,p,t,gh:(0.0,      pMaxQInHP        [gh]), doc='heat input  heat pump                         [MW]')
mBTC.vQOutHP              = Var(mBTC.ptgh, within=NonNegativeReals, bounds=lambda mBTC,p,t,gh:(0.0, mBTC.pMaxQ            [gh]), doc='heat output heat pump                         [MW]')
mBTC.vPInHP               = Var(mBTC.ptgh, within=NonNegativeReals, bounds=lambda mBTC,p,t,gh:(0.0,      pMaxPInHP        [gh]), doc='power input heat pump                         [MW]')
mBTC.vQutility            = Var(mBTC.ptgu, within=NonNegativeReals, bounds=lambda mBTC,p,t,gu:(0.0,      pMaxQ            [gu]), doc='utility heat output                           [MW]')
mBTC.vCommitment          = Var(mBTC.ptg , within=Binary          , initialize=                0                               , doc='commitment of the unit during t (1 if up)    {0,1}')
mBTC.vStartUp             = Var(mBTC.ptg , within=UnitInterval    , initialize=                0.0                             , doc='start-up   of the unit (1 if it starts in t) [0,1]')
mBTC.vShutDown            = Var(mBTC.ptg , within=UnitInterval    , initialize=                0.0                             , doc='shut-down  of the unit (1 if is shuts  in t) [0,1]')
mBTC.vSUType              = Var(mBTC.ptgl, within=UnitInterval    , initialize=                0.0                             , doc='start-up type l in t   (1 if it starts in t) [0,1]')
mBTC.vOnlineStates        = Var(mBTC.ptg , within=UnitInterval    , initialize=                0.0                             , doc='online states of unit g                      [0,1]')
mBTC.vPowerBuy            = Var(mBTC.pt  , within=NonNegativeReals, bounds=lambda mBTC,p,t   :(0.0, mBTC.pGCPCapacity         ), doc='power buy                                     [MW]')
mBTC.vPowerSell           = Var(mBTC.pt  , within=NonNegativeReals, bounds=lambda mBTC,p,t   :(0.0, mBTC.pGCPCapacity         ), doc='power sell                                    [MW]')
mBTC.vContractedPower     = Var(mBTC.pm  , within=NonNegativeReals, bounds=lambda mBTC,p,m   :(0.0, mBTC.pGCPCapacity         ), doc='contracted power per market period            [MW]')
mBTC.vQWaste              = Var(mBTC.ptgg, within=NonNegativeReals, bounds=lambda mBTC,p,t,gg:(0.0, mBTC.pMaxQ            [gg]), doc='waste produced heat                          [MW]')
mBTC.vFuel                = Var(mBTC.ptg , within=NonNegativeReals                                                             , doc='fuel consumption flow                        [MWh]')
mBTC.vRevenue             = Var(mBTC.pt  , within=NonNegativeReals                                                             , doc='hourly system      revenue                   [EUR]')
mBTC.vOpCost              = Var(mBTC.pt  , within=NonNegativeReals                                                             , doc='hourly operational cost                      [EUR]')
mBTC.vTotalBillCost       = Var(           within=NonNegativeReals                                                             , doc='total  electricity cost                      [EUR]')
mBTC.vTotalProfit         = Var(           within=           Reals                                                             , doc='total  system      profit                    [EUR]')
mBTC.vTotalCost           = Var(           within=NonNegativeReals                                                             , doc='total  system      cost                      [EUR]')
mBTC.vTotalRevenue        = Var(           within=NonNegativeReals                                                             , doc='total  system      revenue                   [EUR]')

nFixedVariables = 0.0

for p,t,gc in mBTC.ptgc:
    if mBTC.pIndOperReserve[gc] != 1:
        mBTC.vReserveUp     [p,t,gc].fix(0.0)
        mBTC.vReserveDown   [p,t,gc].fix(0.0)
        mBTC.vActivationUp  [p,t,gc].fix(0.0)
        mBTC.vActivationDown[p,t,gc].fix(0.0)
        mBTC.vActivation    [p,t,gc].fix(0)
        nFixedVariables += 5

if sum(mBTC.pHPOperation[gc] for gc in mBTC.gc) == 0:
    for p,t,gh in mBTC.ptgh:
        mBTC.vQHP       [p,t,gh].fix(0.0)
        mBTC.vQInHP     [p,t,gh].fix(0.0)
        mBTC.vQOutHP    [p,t,gh].fix(0.0)
        mBTC.vPInHP     [p,t,gh].fix(0.0)
        mBTC.vQFS       [p,t,gh].fix(0.0)
        mBTC.vQWaste    [p,t,gh].fix(0.0)
        nFixedVariables += 6
    for p,t,gu in mBTC.ptgu:
        mBTC.vQutility  [p,t,gu].fix(0.0)
        nFixedVariables += 1

for p,t,gc in mBTC.ptgc:
    if mBTC.pHPOperation[gc] == 1:
        mBTC.vQFS           [p,t,gc].fix(0.0)
        nFixedVariables += 1

for p,t,gu in mBTC.ptgu:
    if mBTC.pIndHPUtility[gu] == 1:
        mBTC.vQFS           [p,t,gu].fix(0.0)
        nFixedVariables += 1

for p,t,gh in mBTC.ptgh:
    if mBTC.pQDemand[p,t] == 0.0:
        mBTC.vQOutHP        [p,t,gh].fix(0.0)
        nFixedVariables += 1

mBTC.nFixedVariables = Param(initialize=round(nFixedVariables), within=NonNegativeIntegers, doc='Number of fixed variables')

for p,t,gc in mBTC.ptgc:
    if mBTC.pHPOperation[gc] != 1:
        mBTC.vPFS[p,t,gc].setlb(0.0)
        [mBTC.vPFS[p,t,gc].setub(mBTC.pMaxP[gc]) for gc in mBTC.gc]


SettingUpDataTime = time.time() - StartTime
StartTime         = time.time()
print('Setting up input data.................... ', round(SettingUpDataTime), 's')


#%% Mathematical Formulation

#Start-up type

def eStartUpType1(mBTC,p,t,gcb,lc):
    if (gcb,lc) != mBTC.l2b.last() and mBTC.t.ord(t) >= pOffDuration[gcb, mBTC.lc.next(lc)] and (gcb,lc) in mBTC.l2b:
        return mBTC.vSUType[p,t,gcb,lc] <= sum(mBTC.vShutDown[p,(t-i),gcb] for i in range(pOffDuration[gcb,lc],pOffDuration[gcb,mBTC.lc.next(lc)]))
    else:
        return Constraint.Skip
mBTC.eStartUpType1 = Constraint(mBTC.ptgblc      , rule=eStartUpType1,      doc='start-up type (1a)')

def eStartUpType2(mBTC,p,t,gcs,lc):
    if (gcs,lc) != mBTC.l2s.last() and mBTC.t.ord(t) >= pOffDuration[gcs, mBTC.lc.next(lc)] and (gcs,lc) in mBTC.l2s:
        return mBTC.vSUType[p,t,gcs,lc] <= sum(mBTC.vShutDown[p,(t-i),gcs] for i in range(pOffDuration[gcs,lc],pOffDuration[gcs,mBTC.lc.next(lc)]))
    else:
        return Constraint.Skip
mBTC.eStartUpType2 = Constraint(mBTC.ptgslc      , rule=eStartUpType2,      doc='start-up type (1b)')


#Only one SU type is selected when the unit starts up

def eUnitStartUp(mBTC,p,t,gc):
    return sum(mBTC.vSUType[p,t,gc,lc] for lc in mBTC.lc if (gc,lc) in mBTC.l2c) == mBTC.vStartUp[p,t,gc]
mBTC.eUnitStartUp  = Constraint(mBTC.ptgc        , rule=eUnitStartUp,      doc='only one SU type is selected (2)')


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

def eMaxPPowerLimit(mBTC,p,t,gc):
    if t < mBTC.t.last():
        return (mBTC.vP[p,t,gc] + mBTC.vReserveUp  [p,t,gc]) / mBTC.pMaxP2ndBlock[gc] <= mBTC.vCommitment[p,t,gc] - mBTC.vShutDown[p,mBTC.t.next(t),gc]
    else:
        return Constraint.Skip
mBTC.eMaxPPowerLimit = Constraint(mBTC.ptgc      , rule=eMaxPPowerLimit,    doc='max electric power limit')

def eMinPPowerLimit(mBTC,p,t,gc):
    return     (mBTC.vP[p,t,gc] - mBTC.vReserveDown[p,t,gc]) / mBTC.pMaxP2ndBlock[gc] >= 0.0
mBTC.eMinPPowerLimit = Constraint(mBTC.ptgc      , rule=eMinPPowerLimit,    doc='min electric power limit')

def eMaxQPowerLimit(mBTC,p,t,gc):
    if t < mBTC.t.last():
        return  mBTC.vQ[p,t,gc]                              / mBTC.pMaxQ2ndBlock[gc] <= mBTC.vCommitment[p,t,gc] - mBTC.vShutDown[p,mBTC.t.next(t),gc]
    else:
        return Constraint.Skip
mBTC.eMaxQPowerLimit = Constraint(mBTC.ptgc      , rule=eMaxQPowerLimit,    doc='max thermal power limit')


#Operating Ramp Constraints

def eOperatingRampUp(mBTC,p,t,gc):
    if t > mBTC.t.first():
        return ((mBTC.vP[p,t,gc] + mBTC.vReserveUp  [p,t,gc] - mBTC.vP[p,mBTC.t.prev(t),gc] - mBTC.vReserveDown[p,mBTC.t.prev(t),gc]) / (mBTC.pDuration[t] * mBTC.pRampUp[gc])) <=  mBTC.vCommitment             [p,t,gc] - mBTC.vStartUp [p,t,gc]
    else:
        return Constraint.Skip
mBTC.eOperatingRampUp = Constraint(mBTC.ptgc     , rule=eOperatingRampUp,   doc='operating ramp up')

def eOperatingRampDw(mBTC,p,t,gc):
    if t > mBTC.t.first():
        return ((mBTC.vP[p,t,gc] + mBTC.vReserveDown[p,t,gc] - mBTC.vP[p,mBTC.t.prev(t),gc] - mBTC.vReserveUp  [p,mBTC.t.prev(t),gc]) / (mBTC.pDuration[t] * mBTC.pRampDw[gc])) >= -mBTC.vCommitment[p,mBTC.t.prev(t),gc] + mBTC.vShutDown[p,t,gc]
    else:
        return Constraint.Skip
mBTC.eOperatingRampDw = Constraint(mBTC.ptgc     , rule=eOperatingRampDw,   doc='operating ramp dw')


#OperatingReserves

def eReserveMinRatioDwUp(mBTC,p,t,gc):
    return mBTC.vReserveDown[p,t,gc] >= mBTC.vReserveUp[p,t,gc] * mBTC.pMinRatioDwUp
mBTC.eReserveMinRatioDwUp = Constraint(mBTC.ptgc , rule=eReserveMinRatioDwUp, doc='minimum ratio down to up operating reserve [MW]')

def eReserveMaxRatioDwUp(mBTC,p,t,gc):
    return mBTC.vReserveDown[p,t,gc] <= mBTC.vReserveUp[p,t,gc] * mBTC.pMaxRatioDwUp
mBTC.eReserveMaxRatioDwUp = Constraint(mBTC.ptgc , rule=eReserveMaxRatioDwUp, doc='maximum ratio down to up operating reserve [MW]')

def eActivationUp(mBTC,p,t,gc):
    return mBTC.vActivationUp[p,t,gc]   == mBTC.vReserveUp  [p,t,gc] * mBTC.pActUp[p,t]
mBTC.eActivationUp = Constraint(mBTC.ptgc        , rule=eActivationUp,      doc='relationship between activation and availability up [MWh]')

def eActivationDw(mBTC,p,t,gc):
    return mBTC.vActivationDown[p,t,gc] == mBTC.vReserveDown[p,t,gc] * mBTC.pActDw[p,t]
mBTC.eActivationDw = Constraint(mBTC.ptgc        , rule=eActivationDw,      doc='relationship between activation and availability dw [MWh]')

def eMaxActivationUp(mBTC,p,t,gc):
    return mBTC.vActivationUp[p,t,gc]      <=      mBTC.vActivation[p,t,gc] * mBTC.pDuration[t]  * mBTC.pMaxP2ndBlock[gc]
mBTC.eMaxActivationUp = Constraint(mBTC.ptgc     , rule=eMaxActivationUp,   doc='max activation up [MWh]')

def eMaxActivationDw(mBTC,p,t,gc):
    return mBTC.vActivationDown[p,t,gc]    <= (1 - mBTC.vActivation[p,t,gc]) * mBTC.pDuration[t] * mBTC.pMaxP2ndBlock[gc]
mBTC.eMaxActivationDw = Constraint(mBTC.ptgc     , rule=eMaxActivationDw,   doc='max activation dw [MWh]')

def eMinActivationDw(mBTC,p,t,gc):
    if mBTC.pIndOperReserve[gc] == 1:
        return mBTC.vActivationDown[p,t,gc] >= (1 - mBTC.vActivation[p,t,gc]) * mBTC.pDuration[t] * 0.001
    else:
        return Constraint.Skip
mBTC.eMinActivationDw = Constraint(mBTC.ptgc     , rule=eMinActivationDw,   doc='min activation dw [MWh]')


#PowerSchedule

def eTotalPOutputCHP(mBTC,p,t,gc):
    return mBTC.vTotalOutput[p,t,gc] == mBTC.vPOutput[p,t,gc] + mBTC.vQOutput[p,t,gc]
mBTC.eTotalPOutputCHP = Constraint(mBTC.ptgc     , rule=eTotalPOutputCHP,   doc='CHP total power output')

def eOnlineStatesCHP(mBTC,p,t,gc):
    if mBTC.t.first() < t < (mBTC.t.last() - pMaxSUDuration[gc]):
        return mBTC.vOnlineStates[p,t,gc] == (mBTC.vCommitment[p,t,gc] +
                                             sum(mBTC.vShutDown[p,(t-i+1),gc] for i in range(1, int(pSDDuration[gc])+1)) +
                                             sum(sum(mBTC.vSUType[p,(t-j+pSUDuration[gc,lc]+1),gc,lc] for j in range(1, int(pSUDuration[gc,lc])+1)) for lc in mBTC.lc if (gc,lc) in mBTC.l2c))
    elif (mBTC.t.last() - pMaxSUDuration[gc]) <= t <= mBTC.t.last():
        return mBTC.vOnlineStates[p,t,gc] == (mBTC.vCommitment[p,t,gc] +
                                             sum(mBTC.vShutDown[p,(t-i+1),gc] for i in range(1, int(pSDDuration[gc])+1)))
    else:
        return Constraint.Skip
mBTC.eOnlineStatesCHP = Constraint(mBTC.ptgc     , rule=eOnlineStatesCHP,   doc='CHP online states')

def eEnergyProduction(mBTC,p,t,g):
    return mBTC.vEnergy[p,t,g] == mBTC.vTotalOutput[p,t,g] * mBTC.pDuration[t]
mBTC.eEnergyProduction = Constraint(mBTC.ptg     , rule=eEnergyProduction,  doc='energy production (8)')

def eFuelFlow(mBTC,p,t,g):
    return mBTC.vFuel[p,t,g] == mBTC.vEnergy[p,t,g] / mBTC.pEfficiency[g]
mBTC.eFuelCost = Constraint(mBTC.ptg,              rule=eFuelFlow,          doc='fuel flow')


#PowerBalance

def ePOutput(mBTC,p,t,gc):
    if t < (mBTC.t.last() - pMaxSUDuration[gc]):
        return mBTC.vPOutput[p,t,gc] == ((mBTC.pMinP[gc] * (mBTC.vCommitment[p,t,gc] + mBTC.vStartUp[p,mBTC.t.next(t),gc])) + mBTC.vP[p,t,gc] +
                                        ((mBTC.vActivationUp[p,t,gc] - mBTC.vActivationDown[p,t,gc]) / mBTC.pDuration[t])                     +
                                        sum((mBTC.pPsdi[gc,i] * mBTC.vShutDown[p,(t-i+2),gc]) for i in range(2, int(pSDDuration[gc])+2))      +
                                        sum(sum((mBTC.pPsui[gc,lc,i] * mBTC.vSUType[p,(t-i+pSUDuration[gc,lc]+2),gc,lc]) for i in range(1,pSUDuration[gc,lc]+1)) for lc in mBTC.lc if (gc,lc) in mBTC.l2c))
    elif   (mBTC.t.last() - pMaxSUDuration[gc]) <= t <= mBTC.t.last():
        return mBTC.vPOutput[p,t,gc] == ((mBTC.pMinP[gc] * mBTC.vCommitment[p,t,gc])                                        + mBTC.vP[p,t,gc] +
                                        ((mBTC.vActivationUp[p,t,gc] - mBTC.vActivationDown[p,t,gc]) / mBTC.pDuration[t])                     +
                                        sum((mBTC.pPsdi[gc,i] * mBTC.vShutDown[p,(t-i+2),gc]) for i in range(2, int(pSDDuration[gc])+2))      )
    else:
        return Constraint.Skip
mBTC.ePOutput = Constraint(mBTC.ptgc             , rule=ePOutput,           doc='CHP electric power balance [MW]')

def eQOutput(mBTC,p,t,gc):
    return mBTC.vQOutput[p,t,gc] == (mBTC.pMinQ[gc] * mBTC.vCommitment[p,t,gc]) + mBTC.vQ[p,t,gc]
mBTC.eQOutput = Constraint(mBTC.ptgc             , rule=eQOutput,           doc='CHP thermal power balance [MW]')

def ePQCurve(mBTC,p,t,gc):
    return mBTC.vP[p,t,gc] + ((mBTC.vActivationUp[p,t,gc] - mBTC.vActivationDown[p,t,gc]) / mBTC.pDuration[t]) == mBTC.pPQSlope[gc] * mBTC.vQ[p,t,gc]
mBTC.ePQCurve = Constraint(mBTC.ptgc             , rule=ePQCurve,           doc='BTC PQ relation [MW]')

def eBTCBalance2(mBTC,p,t,gc):
    if mBTC.pHPOperation[gc] == 1:
        return mBTC.vPOutput[p,t,gc] == mBTC.vPFS[p,t,gc] + sum(mBTC.vPInHP[p,t,gh] for gh in mBTC.gh if gh not in mBTC.gu) + ((mBTC.vActivationUp[p,t,gc] - mBTC.vActivationDown[p,t,gc]) / mBTC.pDuration[t])
    else:
        return mBTC.vPOutput[p,t,gc] == mBTC.vPFS[p,t,gc]                                                                   + ((mBTC.vActivationUp[p,t,gc] - mBTC.vActivationDown[p,t,gc]) / mBTC.pDuration[t])
mBTC.eBTCBalance2 = Constraint(mBTC.ptgc         , rule=eBTCBalance2,       doc='BTC output balance 2 [MW]')

def eBTCBalance3(mBTC,p,t,gc):
    if mBTC.pHPOperation[gc] == 1:
        return mBTC.vQOutput[p,t,gc] == sum(mBTC.vQInHP[p,t,gh] for gh in mBTC.gh) + mBTC.vQWaste[p,t,gc]
    else:
        return mBTC.vQOutput[p,t,gc] ==     mBTC.vQFS  [p,t,gc]                    + mBTC.vQWaste[p,t,gc]
mBTC.eBTCBalance3 = Constraint(mBTC.ptgc         , rule=eBTCBalance3,       doc='BTC output balance 3 [MW]')

def eHPBalance(mBTC,p,t,gh):
    if sum(mBTC.pHPOperation[gc] for gc in mBTC.gc) >= 1:
        return mBTC.vQOutHP[p,t,gh] == (pMinPower[gh] * sum(mBTC.vCommitment[p,t,gc] for gc in mBTC.gc if mBTC.pHPOperation[gc] == 1)) + mBTC.vQHP[p,t,gh]
    else:
        return Constraint.Skip
mBTC.eHPBalance = Constraint(mBTC.ptgh           , rule=eHPBalance,         doc='heat pump balance [MW]')

def eHPBalance1(mBTC,p,t,gh):
    if sum(mBTC.pHPOperation[gc] for gc in mBTC.gc) >= 1:
        return mBTC.vQInHP[p,t,gh] == mBTC.vQOutHP[p,t,gh] * (1 - (1 / mBTC.pCOP[gh]))
    else:
        return Constraint.Skip
mBTC.eHPBalance1 = Constraint(mBTC.ptgh          , rule=eHPBalance1,        doc='heat pump balance 1 [MW]')

def eHPBalance2(mBTC,p,t,gh):
    if sum(mBTC.pHPOperation[gc] for gc in mBTC.gc) >= 1:
        return mBTC.vPInHP[p,t,gh] == mBTC.vQOutHP[p,t,gh] - mBTC.vQInHP[p,t,gh]
    else:
        return Constraint.Skip
mBTC.eHPBalance2 = Constraint(mBTC.ptgh          , rule=eHPBalance2,        doc='heat pump balance 2 [MW]')

def eHPBalance3(mBTC,p,t,gh):
    if sum(mBTC.pHPOperation[gc] for gc in mBTC.gc) >= 1:
        if mBTC.pIndHPUtility[gh] == 0:
            return mBTC.vQOutHP[p,t,gh] == mBTC.vQFS     [p,t,gh] + mBTC.vQWaste[p,t,gh]
        else:
            return mBTC.vQOutHP[p,t,gh] == mBTC.vQutility[p,t,gh] + mBTC.vQWaste[p,t,gh]
    else:
        return Constraint.Skip
mBTC.eHPBalance3 = Constraint(mBTC.ptgh          , rule=eHPBalance3,        doc='heat pump balance 3 [MW]')

def ePBalance(mBTC,p,t):
    return sum(mBTC.vPFS[p,t,gc] for gc in mBTC.gc) + mBTC.vPowerBuy[p,t] == mBTC.pPDemand[p,t] + mBTC.vPowerSell[p,t] + sum(mBTC.vPInHP[p,t,gu] for gu in mBTC.gu)
mBTC.ePBalance = Constraint(mBTC.pt              , rule=ePBalance,          doc='electric power balance [MW]')

def eGridConnection(mBTC,p,t):
    return mBTC.vPowerBuy[p,t] + mBTC.vPowerSell[p,t] <= mBTC.pGCPCapacity
mBTC.eGridConnection = Constraint(mBTC.pt        , rule=eGridConnection,    doc='grid connection limit [MW]')

def ePurchase(mBTC,p,t):
    return mBTC.vPowerBuy[p,t] <= mBTC.pPDemand[p,t]
mBTC.ePurchase = Constraint(mBTC.pt             , rule=ePurchase,           doc='maximum purchase [MW]')

def eSale(mBTC,p,t):
    return mBTC.vPowerSell[p,t] <= sum(mBTC.vPFS[p,t,gc] for gc in mBTC.gc)
mBTC.eSale     = Constraint(mBTC.pt              , rule=eSale,              doc='maximum sale [MW]')

def eQBalance(mBTC,p,t):
    return (sum(mBTC.vQFS[p,t,gg] for gg in mBTC.gg)) == mBTC.pQDemand[p,t]
mBTC.eQBalance = Constraint(mBTC.pt              , rule=eQBalance,          doc='thermal power balance [MW]')


#Contracted power by market period

def eContractedPower(mBTC,p,t,m):
    return mBTC.vPowerBuy[p,t] * mBTC.pMarketPeriods[p,t,m] <= mBTC.vContractedPower[p,m]
mBTC.eContractedPower  = Constraint(mBTC.ptm     , rule=eContractedPower,   doc='contracted power per market period [MW]')

def ePowerRule(mBTC,p,m):
    if m > mBTC.m.first():
        return mBTC.vContractedPower[p,mBTC.m.prev(m)]      <= mBTC.vContractedPower[p,m]
    else:
        return Constraint.Skip
mBTC.ePowerRule = Constraint(mBTC.pm             , rule=ePowerRule,         doc='contracted power rule [MW]')


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

def eIniSUType2(mBTC,p,t,gc,lc):
    if t <= pSUDuration[gc,lc] and (gc,lc) in mBTC.l2c:
        return mBTC.vSUType[p,t,gc,lc] == 0.0
    else:
        return Constraint.Skip
mBTC.eIniSUType2 = Constraint(mBTC.ptgclc        , rule=eIniSUType2,        doc='initial su type (15b)')


#%% Objective Function

def eRevenue(mBTC,p,t):
    return mBTC.vRevenue[p,t] ==     ((mBTC.pEnergyPrice[p,t] * mBTC.pDuration[t] * mBTC.vPowerSell      [p,t   ])                    +
                                  sum((mBTC.pSteamPrice [p,t] * mBTC.pDuration[t] * mBTC.vQutility       [p,t,gu]) for gu in mBTC.gu) +
                                  sum((mBTC.pAvaPrice   [p,t]                     * mBTC.vReserveUp      [p,t,gc])                    +
                                      (mBTC.pAvaPrice   [p,t]                     * mBTC.vReserveDown    [p,t,gc]) for gc in mBTC.gc) +
                                  sum((mBTC.pActUpPrice [p,t] * mBTC.pDuration[t] * mBTC.vActivationUp   [p,t,gc])                    +
                                      (mBTC.pActDwPrice [p,t] * mBTC.pDuration[t] * mBTC.vActivationDown [p,t,gc]) for gc in mBTC.gc) )
mBTC.eRevenue = Constraint(mBTC.pt               , rule=eRevenue,           doc='hourly system revenue [EUR]')

def eOpCost(mBTC,p,t):
    return mBTC.vOpCost[p,t] == (sum(((mBTC.pFuelCost            [p,t,g]                      * mBTC.vFuel       [p,t,g    ])                           +
                                 sum( (mBTC.pStartUpCost         [gc,lc]                      * mBTC.vSUType     [p,t,gc,lc])  for (gc,lc) in mBTC.l2c) +
                                      (mBTC.pShutDownCost        [g    ]                      * mBTC.vShutDown   [p,t,g    ])                           +
                                    ( (mBTC.pCO2Cost * mBTC.pCO2ERate[g]) * mBTC.pDuration[t] * mBTC.vTotalOutput[p,t,g    ])) for  g      in mBTC.g  ) +
                                 sum( (mBTC.pCoolingCost                  * mBTC.pDuration[t] * mBTC.vQWaste     [p,t,gg   ])  for  gg     in mBTC.gg ) )
mBTC.eOpCost = Constraint(mBTC.pt                , rule=eOpCost,            doc='hourly operational cost [EUR]')

def eTotalBillCost(mBTC):
    return mBTC.vTotalBillCost == ((sum(mBTC.pEnergyCost[p,t] * mBTC.pDuration[t] * mBTC.vPowerBuy       [p,t] for p,t in mBTC.pt) +
                                    sum(mBTC.pPowerCost [p,m] * 1e3               * mBTC.vContractedPower[p,m] for p,m in mBTC.pm)) * (1.0 + mBTC.pVAT + mBTC.pSET))
mBTC.eTotalBillCost = Constraint(                  rule=eTotalBillCost,     doc='total electricit bill cost [EUR]')

def eTotalRevenue(mBTC):
    return mBTC.vTotalRevenue == sum(mBTC.vRevenue[p,t] for p,t in mBTC.pt)
mBTC.eTotalRevenue = Constraint(                   rule=eTotalRevenue,      doc='total system revenue [EUR]')

def eTotalCost(mBTC):
    return mBTC.vTotalCost   == sum(mBTC.vOpCost  [p,t] for p,t in mBTC.pt) + mBTC.vTotalBillCost
mBTC.eTotalCost = Constraint(                      rule=eTotalCost,         doc='total system cost [EUR]')

def eTotalProfit(mBTC):
    return mBTC.vTotalProfit == mBTC.vTotalRevenue - mBTC.vTotalCost
mBTC.eTotalProfit = Constraint(                    rule=eTotalProfit,       doc='total system profit [EUR]')

def eObjFunction(mBTC):
    return mBTC.vTotalProfit
mBTC.eObjFunction = Objective(rule=eObjFunction  , sense=maximize,          doc='objective function [EUR]')


GeneratingOFTime = time.time() - StartTime
StartTime        = time.time()
print('Generating objective function............ ', round(GeneratingOFTime), 's')


#%% Problem solving

mBTC.write(_path+'/BTC_'+CaseName+'.lp', io_options={'symbolic_solver_labels': True})   # create lp-format file
Solver = SolverFactory(SolverName)                                                      # select solver
Solver.options['LogFile'       ] = _path+'/BTC_'+CaseName+'.log'
Solver.options['OutputFlag'    ] = 1
Solver.options['IISFile'       ] = _path+'/BTC_'+CaseName+'.ilp'                        # should be uncommented to show results of IIS
Solver.options['Method'        ] = 2                                                    # barrier method
Solver.options['MIPGap'        ] = 0.01
Solver.options['Threads'       ] = int((psutil.cpu_count(logical=True) + psutil.cpu_count(logical=False))/2)
Solver.options['TimeLimit'     ] = 7200
Solver.options['IterationLimit'] = 7200000
SolverResults = Solver.solve(mBTC, tee=False)                                           # tee=True displays the output of the solver
SolverResults.write()                                                                   # summary of the solver results

for p,t,g in mBTC.ptg:
    mBTC.vCommitment  [p,t,g].fix(round(mBTC.vCommitment  [p,t,g]()))
    mBTC.vStartUp     [p,t,g].fix(round(mBTC.vStartUp     [p,t,g]()))
    mBTC.vShutDown    [p,t,g].fix(round(mBTC.vShutDown    [p,t,g]()))
    mBTC.vOnlineStates[p,t,g].fix(round(mBTC.vOnlineStates[p,t,g]()))

for p,t,gc in mBTC.ptgc:
    mBTC.vActivation  [p,t,gc].fix(round(mBTC.vActivation [p,t,gc]()))

for p,t,g,l in mBTC.ptgl:
    mBTC.vSUType    [p,t,g,l].fix(round(mBTC.vSUType     [p,t,g,l]()))

Solver.options['relax_integrality'] = 1                                                 # introduced to show results of the dual variables
mBTC.dual = Suffix(direction=Suffix.IMPORT)
SolverResults = Solver.solve(mBTC, tee=False)                                           # tee=True displays the output of the solver
SolverResults.write()                                                                   # summary of the solver results

SolvingTime = time.time() - StartTime
StartTime   = time.time()

print('***** Period: ' + str(p) + ' ******')
print('Problem size............................. ', mBTC.model().nconstraints(), 'constraints, ', mBTC.model().nvariables() - mBTC.nFixedVariables + 1, 'variables')
print('Solution time............................ ', round(SolvingTime), 's')
print('Total system revenues.................... ', round(mBTC.vTotalRevenue() *1e-6, ndigits=2), '[MEUR]')
print('Total system costs....................... ', round(mBTC.vTotalCost()    *1e-6, ndigits=2), '[MEUR]')
print('Total system profit...................... ', round(mBTC.vTotalProfit()  *1e-6, ndigits=2), '[MEUR]')


#%% Final Power Schedule

OfflineStates  = pd.Series(data=[1.0 - mBTC.vOnlineStates[p,t,g]() for p,t,g in mBTC.ptg], index=pd.MultiIndex.from_tuples(mBTC.ptg))

#%% Total Costs by type

ElectricityCost =  sum(mBTC.pEnergyCost  [p,t  ]         * pDuration[t] * mBTC.vPowerBuy       [p,t      ]() for p,t       in mBTC.pt      )
PowerCost       =  sum(mBTC.pPowerCost   [p,m  ]         * 1e3          * mBTC.vContractedPower[p,m      ]() for p,m       in mBTC.pm      )
FuelCost        =  sum(mBTC.pFuelCost    [p,t,g]                        * mBTC.vFuel           [p,t,g    ]() for p,t,g     in mBTC.ptg     )
EmissionCost    =  sum(mBTC.pCO2Cost * mBTC.pCO2ERate[g] * pDuration[t] * mBTC.vTotalOutput    [p,t,g    ]() for p,t,g     in mBTC.ptg     )
WasteHeatCost   =  sum(mBTC.pCoolingCost                 * pDuration[t] * mBTC.vQWaste         [p,t,gg   ]() for p,t,gg    in mBTC.ptgg    )
SUSDCost        = (sum(mBTC.pStartUpCost [gc,lc]                        * mBTC.vSUType         [p,t,gc,lc]() for p,t,gc,lc in mBTC.ptgclc) +
                   sum(mBTC.pShutDownCost[g    ]                        * mBTC.vShutDown       [p,t,g    ]() for p,t,gg    in mBTC.ptgg   ))

#%% Total Revenues by type

PowerSalesRev   =  sum(mBTC.pEnergyPrice [p,t  ]         * pDuration[t] * mBTC.vPowerSell      [p,t      ]() for p,t       in mBTC.pt      )
SteamSalesRev   =  sum(mBTC.pSteamPrice  [p,t  ]         * pDuration[t] * mBTC.vQutility       [p,t,gu   ]() for p,t,gu    in mBTC.ptgu    )
AvaUpRev        =  sum(mBTC.pAvaPrice    [p,t  ]                        * mBTC.vReserveUp      [p,t,gc   ]() for p,t,gc    in mBTC.ptgc    )
AvaDwRev        =  sum(mBTC.pAvaPrice    [p,t  ]                        * mBTC.vReserveDown    [p,t,gc   ]() for p,t,gc    in mBTC.ptgc    )
ActUpRev        =  sum(mBTC.pActUpPrice  [p,t  ]                        * mBTC.vActivationUp   [p,t,gc   ]() for p,t,gc    in mBTC.ptgc    )
ActDwRev        =  sum(mBTC.pActDwPrice  [p,t  ]                        * mBTC.vActivationDown [p,t,gc   ]() for p,t,gc    in mBTC.ptgc    )

#%% Output Results

Out_ElectricityCost = pd.DataFrame({'Electricity Cost [EUR]': [ElectricityCost      *-1]})
Out_PowerCost       = pd.DataFrame({'Cont. Power Cost [EUR]': [PowerCost            *-1]})
Out_BillCost        = pd.DataFrame({'Total Bill Cost [EUR]' : [mBTC.vTotalBillCost()*-1]})
Out_FuelCost        = pd.DataFrame({'Fuel Cost [EUR]'       : [FuelCost             *-1]})
Out_EmissionCost    = pd.DataFrame({'Emission Cost [EUR]'   : [EmissionCost         *-1]})
Out_HeatCost        = pd.DataFrame({'Waste heat Cost [EUR]' : [WasteHeatCost        *-1]})
Out_SUSDCost        = pd.DataFrame({'SU&SD Cost [EUR]'      : [SUSDCost             *-1]})
Output_Costs        = pd.DataFrame({'Total Costs [EUR]'     : [mBTC.vTotalCost()    *-1]})
Out_PSalesRev       = pd.DataFrame({'Power sales Rev [EUR]' : [PowerSalesRev           ]})
Out_SSalesRev       = pd.DataFrame({'Steam sales Rev [EUR]' : [SteamSalesRev           ]})
Out_AvaUpRev        = pd.DataFrame({'UpAva Revenue [EUR]'   : [AvaUpRev                ]})
Out_AvaDwRev        = pd.DataFrame({'DwAva Revenue [EUR]'   : [AvaDwRev                ]})
Out_ActUpRev        = pd.DataFrame({'UpAct Revenue [EUR]'   : [ActUpRev                ]})
Out_ActDwRev        = pd.DataFrame({'DwAct Revenue [EUR]'   : [ActDwRev                ]})
Output_Revenues     = pd.DataFrame({'Total Revenues [EUR]'  : [mBTC.vTotalRevenue()    ]})
Output_Profit       = pd.DataFrame({'Total Profit [EUR]'    : [mBTC.vTotalProfit()     ]})
OutputResults       = pd.concat([Out_ElectricityCost, Out_PowerCost, Out_BillCost, Out_FuelCost, Out_EmissionCost, Out_HeatCost, Out_SUSDCost, Output_Costs, Out_PSalesRev, Out_SSalesRev, Out_AvaUpRev, Out_AvaDwRev, Out_ActUpRev, Out_ActDwRev, Output_Revenues, Output_Profit], axis=1).to_csv(_path + '/BTC_Result_02_CostsSummary_' + CaseName + '.csv', sep=',', index=False)

OfflineStates.to_frame(   name='p.u.').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='p.u.').rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Offline {x}'           ).to_csv(_path+'/BTC_Result_OfflineStates_'+CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vEnergy[p,t,g]()      for p,t,g   in mBTC.ptg],           index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='MWh' ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MWh' ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Energy {x} [MWh]'      ).to_csv(_path+'/BTC_Result_Energy_'       +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vCommitment[p,t,g]()  for p,t,g   in mBTC.ptg],           index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='p.u.').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='p.u.').rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Commitment {x}'        ).to_csv(_path+'/BTC_Result_Commitment_'   +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vStartUp[p,t,g]()     for p,t,g   in mBTC.ptg],           index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='p.u.').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='p.u.').rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'StartUp {x}'           ).to_csv(_path+'/BTC_Result_StartUp_'      +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vShutDown[p,t,g]()    for p,t,g   in mBTC.ptg],           index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='p.u.').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='p.u.').rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ShutDown {x}'          ).to_csv(_path+'/BTC_Result_ShutDown_'     +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vSUType[p,t,g,l]()    for p,t,g,l in mBTC.ptgl],          index=pd.MultiIndex.from_tuples(mBTC.ptgl))
OutputResults.to_frame(   name='p.u.').reset_index().pivot_table( index=['level_0','level_1'], columns='level_3', values='p.u.').rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'SUType {x}'            ).to_csv(_path+'/BTC_Result_SUType_'       +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vTotalOutput[p,t,g]()  for p,t,g  in mBTC.ptg],           index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'TotalOutput {x} [MW]'  ).to_csv(_path+'/BTC_Result_TotalOutput_'  +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vOnlineStates[p,t,g]() for p,t,g in mBTC.ptg],            index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='p.u.').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='p.u.').rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Online {x}'            ).to_csv(_path+'/BTC_Result_OnlineStates_' +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vPowerBuy[p,t]()       for p,t   in mBTC.pt],             index=pd.MultiIndex.from_tuples(mBTC.pt))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'],                    values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'PowerBuy {x}'          ).to_csv(_path+'/BTC_Result_PowerBuy_'     +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vPowerSell[p,t]()      for p,t   in mBTC.pt],             index=pd.MultiIndex.from_tuples(mBTC.pt))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'],                    values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'PowerSell {x}'         ).to_csv(_path+'/BTC_Result_PowerSell_'    +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vPOutput[p,t,gc]()     for p,t,gc in mBTC.ptgc],          index=pd.MultiIndex.from_tuples(mBTC.ptgc))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'POutput {x} [MW]'      ).to_csv(_path+'/BTC_Result_PowerOutput_'  +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vPFS[p,t,gc]()         for p,t,gc in mBTC.ptgc],          index=pd.MultiIndex.from_tuples(mBTC.ptgc))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'PFS {x} [MW]'          ).to_csv(_path+'/BTC_Result_PowerFS_'      +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vQOutput[p,t,g]()      for p,t,g  in mBTC.ptg],           index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QOutput {x} [MW]'      ).to_csv(_path+'/BTC_Result_HeatOutput_'   +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vReserveUp[p,t,gc]()   for p,t,gc in mBTC.ptgc],          index=pd.MultiIndex.from_tuples(mBTC.ptgc))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'AvailabilityUp {x} [MW]').to_csv(_path+'/BTC_Result_ReserveUp_'   +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vReserveDown[p,t,gc]() for p,t,gc in mBTC.ptgc],          index=pd.MultiIndex.from_tuples(mBTC.ptgc))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'AvailabilityDw {x} [MW]').to_csv(_path+'/BTC_Result_ReserveDw_'   +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vActivation[p,t,gc]()  for p,t,gc in mBTC.ptgc],          index=pd.MultiIndex.from_tuples(mBTC.ptgc))
OutputResults.to_frame(   name='-' ).reset_index().pivot_table( index=['level_0','level_1'],   columns='level_2', values='-'   ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Activation {x} [-]'    ).to_csv(_path+'/BTC_Result_Activation_'   +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vActivationUp[p,t,gc]() for p,t,gc in mBTC.ptgc],         index=pd.MultiIndex.from_tuples(mBTC.ptgc))
OutputResults.to_frame(   name='MWh' ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MWh' ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ActivationUp {x} [MWh]').to_csv(_path+'/BTC_Result_ActivationUp_' +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vActivationDown[p,t,gc]() for p,t,gc in mBTC.ptgc],       index=pd.MultiIndex.from_tuples(mBTC.ptgc))
OutputResults.to_frame(   name='MWh' ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MWh' ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ActivationDw {x} [MWh]').to_csv(_path+'/BTC_Result_ActivationDw_' +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vQFS[p,t,gg]()         for p,t,gg in mBTC.ptgg],          index=pd.MultiIndex.from_tuples(mBTC.ptgg))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QFS {x} [MW]'          ).to_csv(_path+'/BTC_Result_HeatFS_'       +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vQWaste[p,t,gg]()      for p,t,gg in mBTC.ptgg],          index=pd.MultiIndex.from_tuples(mBTC.ptgg))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QWaste {x} [MW]'       ).to_csv(_path+'/BTC_Result_HeatWaste_'   +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[(mBTC.pFuelCost[p,t,g]*mBTC.vFuel[p,t,g]()) for p,t,g in mBTC.ptg],index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='EUR' ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='EUR' ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'FuelCost {x} [EUR]'    ).to_csv(_path+'/BTC_Result_FuelCost_'     +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vOpCost[p,t]()           for p,t    in mBTC.pt],          index=pd.MultiIndex.from_tuples(mBTC.pt))
OutputResults.to_frame(   name='EUR' ).reset_index().pivot_table( index=['level_0','level_1'],                   values='EUR'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Cost {x}'              ).to_csv(_path+'/BTC_Result_Costs_'        +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vRevenue[p,t]()         for p,t    in mBTC.pt],           index=pd.MultiIndex.from_tuples(mBTC.pt))
OutputResults.to_frame(   name='EUR' ).reset_index().pivot_table( index=['level_0','level_1'],                   values='EUR'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'Revenue {x}'           ).to_csv(_path+'/BTC_Result_Revenues_'      +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vContractedPower[p,m]() for p,m    in mBTC.pm],           index=pd.MultiIndex.from_tuples(mBTC.pm))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'],                   values='MW' ).rename_axis(['Period','MarketPeriod'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'ContractedP {x}'    ).to_csv(_path+'/BTC_Result_ContractedPower_' +CaseName+'.csv', sep=',')

if sum(mBTC.pHPOperation[gc] for gc in mBTC.gc) >= 1:

    OutputResults = pd.Series(data=[mBTC.vQInHP[p,t,gh]()   for p,t,gh in mBTC.ptgh],          index=pd.MultiIndex.from_tuples(mBTC.ptgh))
    OutputResults.to_frame(   name='MW').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW' ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QInHP {x} [MW]'        ).to_csv(_path+'/BTC_Result_HeatInHP_'     +CaseName+'.csv', sep=',')

    OutputResults = pd.Series(data=[mBTC.vPInHP[p,t,gh]()   for p,t,gh in mBTC.ptgh],          index=pd.MultiIndex.from_tuples(mBTC.ptgh))
    OutputResults.to_frame(   name='MW').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW' ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'PInHP {x} [MW]'        ).to_csv(_path+'/BTC_Result_PowerInHP_'    +CaseName+'.csv', sep=',')

    OutputResults = pd.Series(data=[mBTC.vQOutHP[p,t,gh]()  for p,t,gh in mBTC.ptgh],          index=pd.MultiIndex.from_tuples(mBTC.ptgh))
    OutputResults.to_frame(   name='MW').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW' ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QOutHP {x} [MW]'       ).to_csv(_path+'/BTC_Result_HeatOutHP_'    +CaseName+'.csv', sep=',')

    OutputResults = pd.Series(data=[mBTC.vQutility[p,t,gu]() for p,t,gu in mBTC.ptgu],         index=pd.MultiIndex.from_tuples(mBTC.ptgu))
    OutputResults.to_frame(   name='MW').reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QUtility {x} [MW]'    ).to_csv(_path+'/BTC_Result_UtilitySteam_' +CaseName+'.csv', sep=',')


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
dfActivation        = pd.read_csv(_path + '/BTC_Result_Activation_'     + CaseName + '.csv', index_col=[0, 1])
dfActivationUp      = pd.read_csv(_path + '/BTC_Result_ActivationUp_'   + CaseName + '.csv', index_col=[0, 1])
dfActivationDw      = pd.read_csv(_path + '/BTC_Result_ActivationDw_'   + CaseName + '.csv', index_col=[0, 1])
dfQOutput           = pd.read_csv(_path + '/BTC_Result_HeatOutput_'     + CaseName + '.csv', index_col=[0, 1])
dfQFS               = pd.read_csv(_path + '/BTC_Result_HeatFS_'         + CaseName + '.csv', index_col=[0, 1])
dfQWaste            = pd.read_csv(_path + '/BTC_Result_HeatWaste_'      + CaseName + '.csv', index_col=[0, 1])
dfFuelCost          = pd.read_csv(_path + '/BTC_Result_FuelCost_'       + CaseName + '.csv', index_col=[0, 1])
dfCosts             = pd.read_csv(_path + '/BTC_Result_Costs_'          + CaseName + '.csv', index_col=[0, 1])
dfRevenues          = pd.read_csv(_path + '/BTC_Result_Revenues_'       + CaseName + '.csv', index_col=[0, 1])

if sum(mBTC.pHPOperation[gc] for gc in mBTC.gc) >= 1:
    dfHeatInHP      = pd.read_csv(_path + '/BTC_Result_HeatInHP_'       + CaseName + '.csv', index_col=[0, 1])
    dfPowerInHP     = pd.read_csv(_path + '/BTC_Result_PowerInHP_'      + CaseName + '.csv', index_col=[0, 1])
    dfHeatOutHP     = pd.read_csv(_path + '/BTC_Result_HeatOutHP_'      + CaseName + '.csv', index_col=[0, 1])
    dfQUtility      = pd.read_csv(_path + '/BTC_Result_UtilitySteam_'   + CaseName + '.csv', index_col=[0, 1])

if sum(mBTC.pHPOperation[gc] for gc in mBTC.gc) >= 1:
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
           dfReserveUp,
           dfReserveDw,
           dfActivation,
           dfActivationUp,
           dfActivationDw,
           dfDemand['PDemand'],
           dfQOutput,
           dfHeatInHP,
           dfHeatOutHP,
           dfQWaste,
           dfQFS,
           dfQUtility,
           dfDemand['QDemand'],
           dfCosts,
           dfRevenues]
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
           dfActivation,
           dfActivationUp,
           dfActivationDw,
           dfDemand['PDemand'],
           dfQOutput,
           dfQFS,
           dfQWaste,
           dfDemand['QDemand'],
           dfCosts,
           dfRevenues]

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
