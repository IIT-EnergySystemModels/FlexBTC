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

print('\nProgram for Optimizing the Operation of BTC Plant - Version 1.2.0 - December 14th, 2023')
print('#### Non-commercial use only ####')

for i in range(0, 117):
    print('-', end="")

StartTime = time.time()

DirName    = os.path.dirname(__file__)
CaseName   = 'BTC'
SolverName = 'gurobi'

_path = os.path.join(DirName, CaseName)

#%% Model declaration
mBTC = ConcreteModel('Program for Optimizing the Operation of BTC Plant - Version 1.1.0 - November 15, 2023')

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

#%% general parameters
pPNSCost             = dfParameter  ['PNSCost'             ][0]                # cost of electric energy not served      [EUR/MWh]
pQNSCost             = dfParameter  ['QNSCost'             ][0]                # cost of heat not served                 [EUR/MWh]
pCO2Cost             = dfParameter  ['CO2Cost'             ][0]                # cost of CO2 emission                    [EUR/tonCO2]
pTimeStep            = dfParameter  ['TimeStep'            ][0].astype('int')  # duration of the unit time step          [h]
pAnnualDiscRate      = dfParameter  ['AnnualDiscountRate'  ][0]                # annual discount rate                    [p.u.]
pEconomicBaseYear    = dfParameter  ['EconomicBaseYear'    ][0]                # economic base year                      [year]
pDuration            = dfDuration   ['Duration'            ] * pTimeStep       # duration of load levels                 [h]
pPeriodWeight        = dfPeriod     ['Weight'              ].astype('int')     # weights of periods                      [p.u.]
pPDemand             = dfDemand     ['PDemand'             ]                   # electric power demand                   [MW]
pQDemand             = dfDemand     ['QDemand'             ]                   # heat demand                             [MW]
pEnergyCost          = dfEnergyCost ['Cost'                ]                   # energy cost                             [EUR/MWh]
pEnergyPrice         = dfEnergyPrice['Price'               ]                   # energy price                            [EUR/MWh]
pBiomassCost         = dfFuelCost   ['Biomass'             ]                   # biomass cost                            [EUR/MWh]
pGasCost             = dfFuelCost   ['NaturalGas'          ]                   # natural gas cost                        [EUR/MWh]
pHydrogenCost        = dfFuelCost   ['Hydrogen'            ]                   # hydrogen cost                           [EUR/MWh]
pIndCogeneration     = dfGeneration ['IndCogeneration'     ]                   # generator is a cogeneration unit        [Yes]
pIndFuel             = dfGeneration ['IndFuel'             ]                   # fuel type ('Biomass', 'NaturalGas', or 'Hydrogen')
pMaxPower            = dfGeneration ['MaximumPower'        ]                   # rated maximum power                     [MW]
pMinPower            = dfGeneration ['MinimumPower'        ]                   # rated minimum power                     [MW]
pEfficiency          = dfGeneration ['Efficiency'          ]                   # efficiency                              [p.u.]
pPQSlope             = dfGeneration ['Pqslope'             ]                   # slope of the linear P-Q curve           [MW/MW]
pPQYIntercept        = dfGeneration ['Pqyintercept'        ]                   # y-intercept of the linear P-Q curve     [MW]
pRampUp              = dfGeneration ['RampUp'              ]                   # ramp up   rate                          [MW/h]
pRampDw              = dfGeneration ['RampDown'            ]                   # ramp down rate                          [MW/h]
pUpTime              = dfGeneration ['UpTime'              ]                   # minimum up   time                       [h]
pDwTime              = dfGeneration ['DownTime'            ]                   # minimum down time                       [h]
pShutDownCost        = dfGeneration ['ShutDownCost'        ]                   # shutdown cost                           [EUR]
pSDDuration          = dfGeneration ['SDDuration'          ]                   # duration of the shut-down ramp process  [h]
pCO2EmissionRate     = dfGeneration ['CO2EmissionRate'     ]                   # emission  rate                          [tCO2/MWh]
pNLCost              = dfGeneration ['NLCost'              ]                   # no load cost                            [EUR]
pLVCost              = dfGeneration ['LVCost'              ]                   # linear variable production cost         [EUR/MWh]
pPowerSyn            = dfStartUp    ['PowerSyn'            ]                   # P at which the unit is synchronized     [MW]
pStartUpCost         = dfStartUp    ['StartUpCost'         ]                   # startup  cost                           [EUR]
pSUDuration          = dfStartUp    ['SUDuration'          ]                   # duration of the start-up l ramp process [h]
pOffDuration         = dfStartUp    ['OffDuration'         ]                   # min periods before beginning of SU l    [h]
pCommitment0         = dfInitialCond['Commitment0'         ]                   # initial commitment state of the unit    {0.1}
pUpTime0             = dfInitialCond['UpTime0'             ]                   # nr of hours that has been up before     [h]
pDwTime0             = dfInitialCond['DownTime0'           ]                   # nr of hours that has been dw before     [h]
pOutput0             = dfInitialCond['p0'                  ]                   # initial power output                    [MW]
pPsdi                = dfSDTraject  ['Psd_i'               ]                   # P at beginning of ith interval of SD    [MW]
pPsui                = dfSUTraject  ['Psu_i'               ]                   # P at beginning of ith interval of SU    [MW]

# compute the Demand as the mean over the time step load levels and assign it to active load levels.
pPDemand             = pPDemand.rolling    (pTimeStep).mean()
pQDemand             = pQDemand.rolling    (pTimeStep).mean()
pEnergyCost          = pEnergyCost.rolling (pTimeStep).mean()
pEnergyPrice         = pEnergyPrice.rolling(pTimeStep).mean()

pPDemand.fillna      (0.0, inplace=True)
pQDemand.fillna      (0.0, inplace=True)
pEnergyCost.fillna   (0.0, inplace=True)
pEnergyPrice.fillna  (0.0, inplace=True)
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


# defining subsets
mBTC.p    = Set(initialize=mBTC.pp,         ordered=True , doc='periods'           , filter=lambda mBTC,pp: pp in mBTC.pp and pPeriodWeight[pp] >  0.0                                                   )
mBTC.t    = Set(initialize=mBTC.tt,         ordered=True , doc='load levels'       , filter=lambda mBTC,tt: tt in mBTC.tt and pDuration    [tt] >  0                                                     )
mBTC.t2   = Set(initialize=mBTC.tt,         ordered=True , doc='load levels'       , filter=lambda mBTC,tt: tt in mBTC.tt and pDuration    [tt] >  0                                                     )
mBTC.g    = Set(initialize=mBTC.gg,         ordered=False, doc='generating units'  , filter=lambda mBTC,gg: gg in mBTC.gg and pMaxPower    [gg] >  0.0                                                   )
mBTC.gc   = Set(initialize=mBTC.gg,         ordered=False, doc='cogeneration units', filter=lambda mBTC,gg: gg in mBTC.gg and pMaxPower    [gg] >  0.0 and pIndCogeneration[gg] > 0                      )
mBTC.gx   = Set(initialize=mBTC.gg,         ordered=False, doc='thermal units'     , filter=lambda mBTC,gg: gg in mBTC.gg and pMaxPower    [gg] >  0.0 and pIndCogeneration[gg] < 1 and pNLCost[gg] > 0.0)
mBTC.l    = Set(initialize=mBTC.ll,         ordered=True , doc='su types'          , filter=lambda mBTC,ll: ll in mBTC.ll                                                                                )
mBTC.l2   = Set(initialize=dfStartUp.index, ordered=True , doc='su types index'                                                                                                                          )
mBTC.l2c  = Set(initialize=lambda m: list(mBTC.l2 - {(g, l) for g,l in mBTC.gx*mBTC.l}), ordered=True, doc='BTC su types index'                                                                          )
mBTC.lx   = Set(initialize=lambda m: list(mBTC.l  - {l      for g,l in mBTC.l2c      }), ordered=True, doc='thermal units su types'                                                                      )
mBTC.lc   = Set(initialize=lambda m: list(mBTC.l  - mBTC.lx                           ), ordered=True, doc='BTC su types'                                                                                )
mBTC.sdi  = Set(initialize=lambda m: list(range(1, pSDDuration.max() + 2)             ), ordered=True, doc='shut-down time periods'                                                                      )
mBTC.sui  = Set(initialize=lambda m: list(range(1, pSUDuration.max() + 2)             ), ordered=True, doc='start-up  time periods'                                                                      )

g2l = defaultdict(list)
for g,l in mBTC.l2:
    g2l[l].append(g)

# instrumental sets
mBTC.pg     = [(p, g            ) for p, g             in mBTC.p    * mBTC.g          ]
mBTC.pt     = [(p, t            ) for p, t             in mBTC.p    * mBTC.t          ]
mBTC.ptg    = [(p, t, g         ) for p, t, g          in mBTC.pt   * mBTC.g          ]
mBTC.ptgc   = [(p, t, gc        ) for p, t, gc         in mBTC.pt   * mBTC.gc         ]
mBTC.ptgclc = [(p, t, gc, lc    ) for p, t, gc, lc     in mBTC.ptgc * mBTC.lc         ]
mBTC.ptgx   = [(p, t, gx        ) for p, t, gx         in mBTC.pt   * mBTC.gx         ]
mBTC.ptgxlx = [(p, t, gx, lx    ) for p, t, gx, lx     in mBTC.ptgx * mBTC.lx         ]
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
pDuration     = pDuration.loc    [mBTC.t      ]
pPDemand      = pPDemand.loc     [mBTC.pt     ]
pQDemand      = pQDemand.loc     [mBTC.pt     ]
pEnergyCost   = pEnergyCost.loc  [mBTC.pt     ]
pEnergyPrice  = pEnergyPrice.loc [mBTC.pt     ]

# drop 0 values
pPsui         = pPsui.loc        [pPsui != 0.0]
pPQSlope      = pPQSlope.loc     [mBTC.gc     ]
pPQYIntercept = pPQYIntercept.loc[mBTC.gc     ]

# this option avoids a warning in the following assignments
pd.options.mode.chained_assignment = None

#defining auxiliary parameters
pUpTimeR          = pd.Series(data=[max(0.0,(pUpTime[g] - pUpTime0[g]) *      pCommitment0[g])  for g     in mBTC.g  ], index=mBTC.g )
pDwTimeR          = pd.Series(data=[max(0.0,(pDwTime[g] - pDwTime0[g]) * (1 - pCommitment0[g])) for g     in mBTC.g  ], index=mBTC.g )

for gc, lc in mBTC.gc*mBTC.lc:
    pMaxSUDuration_gc = pSUDuration[gc,lc].max()

for gx, lx in mBTC.gx*mBTC.lx:
    pMaxSUDuration_gx = pSUDuration[gx,lx].max()

pFuelCost = pd.Series(index=mBTC.g, dtype='float64')
for g in mBTC.g:
    if pIndFuel[g] == 'Biomass':
        pFuelCost[g] = pBiomassCost
    elif pIndFuel[g] == 'Hydrogen':
        pFuelCost[g] = pHydrogenCost
    else:
        pFuelCost[g] = pGasCost

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

#Parameters
mBTC.pPNSCost             = Param(             initialize=pPNSCost                      , within=NonNegativeReals,    doc='PNS cost'                              )
mBTC.pQNSCost             = Param(             initialize=pQNSCost                      , within=NonNegativeReals,    doc='QNS cost'                              )
mBTC.pCO2Cost             = Param(             initialize=pCO2Cost                      , within=NonNegativeReals,    doc='CO2 emission cost'                     )
mBTC.pTimeStep            = Param(             initialize=pTimeStep                     , within=PositiveIntegers,    doc='Unitary time step'                     )
mBTC.pAnnualDiscRate      = Param(             initialize=pAnnualDiscRate               , within=UnitInterval,        doc='Annual discount rate'                  )
mBTC.pEconomicBaseYear    = Param(             initialize=pEconomicBaseYear             , within=PositiveIntegers,    doc='Base year'                             )
mBTC.pCommitment0         = Param(mBTC.g     , initialize=pCommitment0.to_dict()        , within=UnitInterval,        doc='Initial commitment'                    )
mBTC.pUpTime0             = Param(mBTC.g     , initialize=pUpTime0.to_dict()            , within=NonNegativeIntegers, doc='Initial Up   time'                     )
mBTC.pDwTime0             = Param(mBTC.g     , initialize=pDwTime0.to_dict()            , within=NonNegativeIntegers, doc='Initial Down time'                     )
mBTC.pOutput0             = Param(mBTC.g     , initialize=pOutput0.to_dict()            , within=NonNegativeIntegers, doc='Initial power output'                  )
mBTC.pUpTimeR             = Param(mBTC.g     , initialize=pUpTimeR.to_dict()            , within=NonNegativeIntegers, doc='Up   time R'                           )
mBTC.pDwTimeR             = Param(mBTC.g     , initialize=pDwTimeR.to_dict()            , within=NonNegativeIntegers, doc='Down time R'                           )

#Generation
mBTC.pIndCogeneration     = Param(mBTC.g     , initialize=pIndCogeneration.to_dict()    , within=UnitInterval,        doc='Indicator of cogeneration unit'        )
mBTC.pMaxPower            = Param(mBTC.g     , initialize=pMaxPower.to_dict()           , within=NonNegativeReals,    doc='Rated maximum power'                   )
mBTC.pMinPower            = Param(mBTC.g     , initialize=pMinPower.to_dict()           , within=NonNegativeReals,    doc='Rated minimum power'                   )
mBTC.pEfficiency          = Param(mBTC.g     , initialize=pEfficiency.to_dict()         , within=UnitInterval,        doc='Round-trip efficiency'                 )
mBTC.pPQSlope             = Param(mBTC.gc    , initialize=pPQSlope.to_dict()            , within=Reals,               doc='Slope of linear PQ curve'              )
mBTC.pPQYIntercept        = Param(mBTC.gc    , initialize=pPQYIntercept .to_dict()      , within=Reals,               doc='Y-Intercept of linear PQ curve'        )
mBTC.pRampUp              = Param(mBTC.g     , initialize=pRampUp.to_dict()             , within=NonNegativeReals,    doc='Ramp up   rate'                        )
mBTC.pRampDw              = Param(mBTC.g     , initialize=pRampDw.to_dict()             , within=NonNegativeReals,    doc='Ramp down rate'                        )
mBTC.pUpTime              = Param(mBTC.g     , initialize=pUpTime.to_dict()             , within=NonNegativeIntegers, doc='Up   time'                             )
mBTC.pDwTime              = Param(mBTC.g     , initialize=pDwTime.to_dict()             , within=NonNegativeIntegers, doc='Down time'                             )
mBTC.pShutDownCost        = Param(mBTC.g     , initialize=pShutDownCost.to_dict()       , within=NonNegativeReals,    doc='Shutdown cost'                         )
mBTC.pSDDuration          = Param(mBTC.g     , initialize=pSDDuration.to_dict()         , within=NonNegativeIntegers, doc='Duration of SD l ramp process'         )
mBTC.pCO2EmissionRate     = Param(mBTC.g     , initialize=pCO2EmissionRate.to_dict()    , within=NonNegativeReals,    doc='Emission Rate'                         )
mBTC.pNLCost              = Param(mBTC.g     , initialize=pNLCost.to_dict()             , within=NonNegativeReals,    doc='No load cost'                          )
mBTC.pLVCost              = Param(mBTC.g     , initialize=pLVCost.to_dict()             , within=NonNegativeReals,    doc='Linear variable cost'                  )

#StartUp
mBTC.pPowerSyn            = Param(mBTC.gl    , initialize=pPowerSyn.to_dict()           , within=NonNegativeReals,    doc='P at which the unit is synchronized'   )
mBTC.pStartUpCost         = Param(mBTC.gl    , initialize=pStartUpCost.to_dict()        , within=NonNegativeReals,    doc='Startup  cost'                         )
mBTC.pSUDuration          = Param(mBTC.gl    , initialize=pSUDuration.to_dict()         , within=NonNegativeIntegers, doc='Duration of SU l ramp process'         )
mBTC.pOffDuration         = Param(mBTC.gl    , initialize=pOffDuration.to_dict()        , within=NonNegativeIntegers, doc='Min periods before beginning of SU l'  )
mBTC.pPsui                = Param(mBTC.glsui , initialize=pPsui.to_dict()               , within=NonNegativeReals,    doc='P at beginning of ith interval of SU'  )

#ShutDown
mBTC.pPsdi                = Param(mBTC.gsdi  , initialize=pPsdi.to_dict()               , within=NonNegativeReals,    doc='P at beginning of ith interval of SD'  )


## Variables

mBTC.vEnergy              = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g:(0.0, mBTC.pMaxPower[g]),  doc='energy production during loadlevel t               [MWh]')
mBTC.vOutput              = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g:(0.0, mBTC.pMaxPower[g]),  doc='output at the end of t, above minimum output        [MW]')
mBTC.vTotalOutput         = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g:(0.0, mBTC.pMaxPower[g]),  doc='total output at the end of t                        [MW]')
mBTC.vPOutput             = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g:(0.0, mBTC.pMaxPower[g]),  doc='power output at the end of t                        [MW]')
mBTC.vQOutput             = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g:(0.0, mBTC.pMaxPower[g]),  doc='heat  output at the end of t                        [MW]')
mBTC.vCommitment          = Var(mBTC.ptg , within=Binary,           initialize=0  ,                                     doc='commitment of the unit during t (1 if up)          {0,1}')
mBTC.vStartUp             = Var(mBTC.ptg , within=UnitInterval,     initialize=0.0,                                     doc='start-up   of the unit (1 if it starts in t)       [0,1]')
mBTC.vShutDown            = Var(mBTC.ptg , within=UnitInterval,     initialize=0.0,                                     doc='shut-down  of the unit (1 if is shuts  in t)       [0,1]')
mBTC.vSUType              = Var(mBTC.ptgl, within=UnitInterval,     initialize=0.0,                                     doc='start-up type l in t   (1 if it starts in t)       [0,1]')
mBTC.vOnlineStates        = Var(mBTC.ptg , within=UnitInterval,     initialize=0.0,                                     doc='online states of unit g                            [0,1]')
mBTC.vPowerBuy            = Var(mBTC.pt  , within=NonNegativeReals, bounds=lambda mBTC,p,t  :(0.0, mBTC.pPDemand[p,t]), doc='power buy                                           [MW]')
mBTC.vPowerSell           = Var(mBTC.pt  , within=NonNegativeReals, bounds=lambda mBTC,p,t  :(0.0, mBTC.pPDemand[p,t]), doc='power sell                                          [MW]')
mBTC.vQExcess             = Var(mBTC.ptg , within=NonNegativeReals, bounds=lambda mBTC,p,t,g:(0.0, mBTC.pMaxPower[g]),  doc='excess produced heat                                [MW]')
mBTC.vFuel                = Var(mBTC.ptg , within=NonNegativeReals,                                                     doc='fuel flow during loadlevel t                       [MWh]')
mBTC.vTotalRevenue        = Var(           within=           Reals,                                                     doc='total system revenue                               [EUR]')
mBTC.vTotalCost           = Var(           within=NonNegativeReals,                                                     doc='total system cost                                  [EUR]')
mBTC.vTotalProfit         = Var(           within=NonNegativeReals,                                                     doc='total system profit                                [EUR]')

nFixedVariables = 0.0
mBTC.nFixedVariables = Param(initialize=round(nFixedVariables), within=NonNegativeIntegers, doc='Number of fixed variables')

SettingUpDataTime = time.time() - StartTime
StartTime         = time.time()
print('Setting up input data.................... ', round(SettingUpDataTime), 's')


#%% Mathematical Formulation

#Start-up type

def eStartUpType1(mBTC,p,t,gc,lc):
    if lc < mBTC.lc.last() and  mBTC.t.ord(t) >= pOffDuration[gc,mBTC.lc.next(lc)]:
        return mBTC.vSUType[p,t,gc,lc] <= sum(mBTC.vShutDown[p,(t-i),gc] for i in range(pOffDuration[gc,lc],(pOffDuration[gc,mBTC.lc.next(lc)]-1)+1))
    else:
        return Constraint.Skip
mBTC.eStartUpType1 = Constraint(mBTC.ptgclc      , rule=eStartUpType1,      doc='start-up type (1a)')

def eStartUpType2(mBTC,p,t,gx,lx):
    if lx < mBTC.lx.last() and  mBTC.t.ord(t) >= pOffDuration[gx,mBTC.lx.next(lx)] and len(mBTC.lx) > 1:
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

def eCapacityLimit(mBTC,p,t,g):
    if t < mBTC.t.last():
        return mBTC.vOutput[p,t,g] <= (mBTC.pMaxPower[g] - mBTC.pMinPower[g]) * (mBTC.vCommitment[p,t,g] - mBTC.vShutDown[p,mBTC.t.next(t),g])
    else:
        return Constraint.Skip
mBTC.eCapacityLimit = Constraint(mBTC.ptg        , rule=eCapacityLimit,     doc='capacity limits (6)')


#Operating Ramp Constraints

def eOperatingRampUp(mBTC,p,t,g):
    if t > mBTC.t.first():
        return (mBTC.vOutput[p,t,g] - mBTC.vOutput[p,mBTC.t.prev(t),g]) <= mBTC.pRampUp[g]
    else:
        return Constraint.Skip
mBTC.eOperatingRampUp = Constraint(mBTC.ptg      , rule=eOperatingRampUp,   doc='operating ramp up (7a)')

def eOperatingRampDw(mBTC,p,t,g):
    if t > mBTC.t.first():
        return (mBTC.vOutput[p,t,g] - mBTC.vOutput[p,mBTC.t.prev(t),g]) >= mBTC.pRampDw[g]*-1
    else:
        return Constraint.Skip
mBTC.eOperatingRampDw = Constraint(mBTC.ptg      , rule=eOperatingRampDw,   doc='operating ramp dw (7b)')


#EnergyProduction

def eEnergyProduction1(mBTC,p,t,gc):
    if mBTC.t.first() < t < (mBTC.t.last() - pMaxSUDuration_gc):
        return mBTC.vEnergy[p,t,gc] == ((mBTC.pMinPower  [gc ] * mBTC.vCommitment[p,t,gc]) +
                                        (mBTC.vOutput[p,t,gc ] + mBTC.vOutput[p,mBTC.t.prev(t),gc]) * 0.5 +
                                        sum((    (mBTC.pPsdi[gc,i   ] + mBTC.pPsdi[gc,   (i+1)])/2) * mBTC.vShutDown[p,(t-i+1),                   gc   ] for i in range(1,pSDDuration[gc   ]+1)) +
                                        sum(sum(((mBTC.pPsui[gc,lc,j] + mBTC.pPsui[gc,lc,(j+1)])/2) * mBTC.vSUType  [p,(t-j+pSUDuration[gc,lc]+1),gc,lc] for j in range(1,pSUDuration[gc,lc]+1)) for gc,lc in mBTC.gc*mBTC.lc))
    elif (mBTC.t.last() - pMaxSUDuration_gc) <= t <= mBTC.t.last():
        return mBTC.vEnergy[p,t,gc] == ((mBTC.pMinPower   [gc] * mBTC.vCommitment[p,t,gc]) +
                                        (mBTC.vOutput[p,t,gc ] + mBTC.vOutput[p,mBTC.t.prev(t),gc]) * 0.5 +
                                        sum(((mBTC.pPsdi[gc,i] + mBTC.pPsdi[gc,(i+1)])/2) * mBTC.vShutDown[p,(t-i+1),gc] for i in range(1,pSDDuration[gc]+1)))
    else:
        return Constraint.Skip
mBTC.eEnergyProduction1 = Constraint(mBTC.ptgc   , rule=eEnergyProduction1, doc='energy production (8a)')

def eEnergyProduction2(mBTC,p,t,gx):
   if mBTC.t.first() < t < (mBTC.t.last() - pMaxSUDuration_gx):
        return mBTC.vEnergy[p,t,gx] == ((mBTC.pMinPower  [gx ] * mBTC.vCommitment[p,t,gx]) +
                                        (mBTC.vOutput[p,t,gx ] + mBTC.vOutput[p,mBTC.t.prev(t),gx]) * 0.5 +
                                        sum((    (mBTC.pPsdi[gx,i   ] + mBTC.pPsdi[gx,   (i+1)])/2) * mBTC.vShutDown[p,(t-i+1),                   gx   ] for i in range(1,pSDDuration[gx   ]+1)) +
                                        sum(sum(((mBTC.pPsui[gx,lx,j] + mBTC.pPsui[gx,lx,(j+1)])/2) * mBTC.vSUType  [p,(t-j+pSUDuration[gx,lx]+1),gx,lx] for j in range(1,pSUDuration[gx,lx]+1)) for gx,lx in mBTC.gx*mBTC.lx))
   elif (mBTC.t.last() - pMaxSUDuration_gx) <= t <= mBTC.t.last():
        return mBTC.vEnergy[p,t,gx] == ((mBTC.pMinPower   [gx] * mBTC.vCommitment[p,t,gx]) +
                                        (mBTC.vOutput[p,t,gx ] + mBTC.vOutput[p,mBTC.t.prev(t),gx]) * 0.5 +
                                        sum(((mBTC.pPsdi[gx,i] + mBTC.pPsdi[gx,(i+1)])/2) * mBTC.vShutDown[p,(t-i+1),gx] for i in range(1,pSDDuration[gx]+1)))
   else:
        return Constraint.Skip
mBTC.eEnergyProduction2 = Constraint(mBTC.ptgx   , rule=eEnergyProduction2, doc='energy production (8b)')


#PowerBalance

def eBTCBalance(mBTC,p,t,gc):
    return mBTC.vTotalOutput[p,t,gc] == mBTC.vPOutput[p,t,gc] + mBTC.vQOutput[p,t,gc]
mBTC.eBTCBalance = Constraint(mBTC.ptgc          , rule=eBTCBalance,        doc='BTC output balance [MW]')

def eXBalance(mBTC,p,t,gx):
    return mBTC.vTotalOutput[p,t,gx] == mBTC.vQOutput[p,t,gx] + mBTC.vQExcess[p,t,gx]
mBTC.eXBalance = Constraint(mBTC.ptgx            , rule=eXBalance,          doc='thermal units output balance [MW]')

def ePQCurve(mBTC,p,t,gc):
    return mBTC.vPOutput[p,t,gc] >= (mBTC.pPQSlope[gc] * mBTC.vQOutput[p,t,gc]) + mBTC.pPQYIntercept[gc]
mBTC.ePQCurve = Constraint(mBTC.ptgc             , rule=ePQCurve,           doc='BTC PQ relation')

def ePBalance(mBTC,p,t):
    return (sum(mBTC.vPOutput[p,t,gc] for gc in mBTC.gc) + mBTC.vPowerBuy[p,t] - mBTC.vPowerSell[p,t]) == mBTC.pPDemand[p,t]
mBTC.ePBalance = Constraint(mBTC.pt              , rule=ePBalance,          doc='electric power balance [MW]')

def eQBalance(mBTC,p,t):
    return sum(mBTC.vQOutput[p,t,g] for g in mBTC.g)                                                   == mBTC.pQDemand[p,t]
mBTC.eQBalance = Constraint(mBTC.pt              , rule=eQBalance,          doc='heat balance [MW]')

def eFuelFlow(mBTC,p,t,g):
    return mBTC.vFuel[p,t,g] == mBTC.vEnergy[p,t,g] / mBTC.pEfficiency[g]
mBTC.eFuelCost = Constraint(mBTC.ptg,              rule=eFuelFlow,          doc='fuel flow')


#Final Power Schedule

def eTotalPOutput1(mBTC,p,t,gc):
    if t < (mBTC.t.last() - pMaxSUDuration_gc):
        return mBTC.vTotalOutput[p,t,gc] == ((mBTC.pMinPower[gc] * (mBTC.vCommitment[p,t,gc] + mBTC.vStartUp[p,mBTC.t.next(t),gc])) + mBTC.vOutput[p,t,gc] +
                                            sum((mBTC.pPsdi[gc,i] * mBTC.vShutDown[p,(t-i+2),gc]) for i in range(2, pSDDuration[gc]+1)) +
                                            sum(sum((mBTC.pPsui[gc,lc,i] * mBTC.vSUType[p,(t-i+pSUDuration[gc,lc]+2),gc,lc]) for i in range(1,pSUDuration[gc,lc]+1)) for gc,lc in mBTC.gc*mBTC.lc))
    elif   (mBTC.t.last() - pMaxSUDuration_gc) <= t <= mBTC.t.last():
        return mBTC.vTotalOutput[p,t,gc] == ((mBTC.pMinPower[gc]  * mBTC.vCommitment[p,t,gc]) + mBTC.vOutput[p,t,gc] +
                                            sum((mBTC.pPsdi[gc,i] * mBTC.vShutDown[p,(t-i+2),gc]) for i in range(2, pSDDuration[gc]+1)))
    else:
        return Constraint.Skip
mBTC.eTotalPOutput1 = Constraint(mBTC.ptgc       , rule=eTotalPOutput1,     doc='total power output (a)')

def eTotalPOutput2(mBTC,p,t,gx):
    if t < (mBTC.t.last() - pMaxSUDuration_gc):
        return mBTC.vTotalOutput[p,t,gx] == ((mBTC.pMinPower[gx] * (mBTC.vCommitment[p,t,gx] + mBTC.vStartUp[p,mBTC.t.next(t),gx])) + mBTC.vOutput[p,t,gx] +
                                            sum((mBTC.pPsdi[gx,i] * mBTC.vShutDown[p,(t-i+2),gx]) for i in range(2, pSDDuration[gx]+1)) +
                                            sum(sum((mBTC.pPsui[gx,lx,i] * mBTC.vSUType[p,(t-i+pSUDuration[gx,lx]+2),gx,lx]) for i in range(1,pSUDuration[gx,lx]+1)) for gx,lx in mBTC.gx*mBTC.lx))
    elif   (mBTC.t.last() - pMaxSUDuration_gx) <= t <= mBTC.t.last():
        return mBTC.vTotalOutput[p,t,gx] == ((mBTC.pMinPower[gx]  * mBTC.vCommitment[p,t,gx]) + mBTC.vOutput[p,t,gx] +
                                            sum((mBTC.pPsdi[gx,i] * mBTC.vShutDown[p,(t-i+2),gx]) for i in range(2, pSDDuration[gx]+1)))
    else:
        return Constraint.Skip
mBTC.eTotalPOutput2 = Constraint(mBTC.ptgx       , rule=eTotalPOutput2,     doc='total power output (b)')


def eOnlineStates1(mBTC,p,t,gc):
    if mBTC.t.first() < t < (mBTC.t.last() - pMaxSUDuration_gc):
        return mBTC.vOnlineStates[p,t,gc] == (mBTC.vCommitment[p,t,gc] +
                                            sum(mBTC.vShutDown[p,(t-i+1),gc] for i in range(1, pSDDuration[gc]+1)) +
                                            sum(sum(mBTC.vSUType[p,(t-j+pSUDuration[gc,lc]+1),gc,lc] for j in range(1, pSUDuration[gc,lc]+1)) for gc,lc in mBTC.gc*mBTC.lc))
    elif (mBTC.t.last() - pMaxSUDuration_gc) <= t <= mBTC.t.last():
        return mBTC.vOnlineStates[p,t,gc] == (mBTC.vCommitment[p,t,gc] +
                                            sum(mBTC.vShutDown[p,(t-i+1),gc] for i in range(1, pSDDuration[gc]+1)))
    else:
        return Constraint.Skip
mBTC.eOnlineStates1 = Constraint(mBTC.ptgc       , rule=eOnlineStates1,     doc='online states (a)')

def eOnlineStates2(mBTC,p,t,gx):
    if mBTC.t.first() < t < (mBTC.t.last() - pMaxSUDuration_gx):
        return mBTC.vOnlineStates[p,t,gx] == (mBTC.vCommitment[p,t,gx] +
                                            sum(mBTC.vShutDown[p,(t-i+1),gx] for i in range(1, pSDDuration[gx]+1)) +
                                            sum(sum(mBTC.vSUType[p,(t-j+pSUDuration[gx,lx]+1),gx,lx] for j in range(1, pSUDuration[gx,lx]+1)) for gx,lx in mBTC.gx*mBTC.lx))
    elif (mBTC.t.last() - pMaxSUDuration_gx) <= t <= mBTC.t.last():
        return mBTC.vOnlineStates[p,t,gx] == (mBTC.vCommitment[p,t,gx] +
                                            sum(mBTC.vShutDown[p,(t-i+1),gx] for i in range(1, pSDDuration[gx]+1)))
    else:
        return Constraint.Skip
mBTC.eOnlineStates2 = Constraint(mBTC.ptgx       , rule=eOnlineStates2,     doc='online states (b)')


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
mBTC.eIniSUType1 = Constraint(mBTC.ptgclc        , rule=eIniSUType1,        doc='initial commitment (15a)')

def eIniSUType2(mBTC,p,t,gx,lx):
    if len(mBTC.lx) > 1 and mBTC.pDwTime0[gx] >= 2.0:
        if lx < mBTC.lx.last() and ((pOffDuration[gx, mBTC.lx.next(lx)] - mBTC.pDwTime0[gx]) < t < pOffDuration[gx, mBTC.lx.next(lx)]):
            return mBTC.vSUType[p,t,gx,lx] == 0.0
        else:
            return Constraint.Skip
    else:
        return Constraint.Skip
mBTC.eIniSUType2 = Constraint(mBTC.ptgxlx        , rule=eIniSUType2,        doc='initial commitment (15b)')


#%% Objective Function

def eTotalProfit(mBTC):
    return mBTC.vTotalProfit == sum((mBTC.pEnergyPrice[p,t] * mBTC.vPowerSell[p,t] * mBTC.pDuration[t]) for p,t in mBTC.pt)
mBTC.eTotalProfit = Constraint(                    rule=eTotalProfit,       doc='total system profit [EUR]')

def eTotalCost(mBTC):
    return mBTC.vTotalCost == (sum(((mBTC.pFuelCost    [g    ] * mBTC.vFuel    [p,t,g    ])                                                            +
                               sum( (mBTC.pStartUpCost [gc,lc] * mBTC.vSUType  [p,t,gc,lc])  for gc,lc in mBTC.gc*mBTC.lc)                             +
                               sum( (mBTC.pStartUpCost [gx,lx] * mBTC.vSUType  [p,t,gx,lx])  for gx,lx in mBTC.gx*mBTC.lx)                             +
                                    (mBTC.pShutDownCost[g    ] * mBTC.vShutDown[p,t,g    ])                                                            +
                                    (mBTC.pCO2Cost * mBTC.pCO2EmissionRate[g] * mBTC.pDuration [t] * mBTC.vTotalOutput[p,t,g])) for p,t,g in mBTC.ptg) +
                               sum((mBTC.pEnergyCost[p,t]                     * mBTC.pDuration [t] * mBTC.vPowerBuy   [p,t  ])  for p,t   in mBTC.pt  ))
mBTC.eTotalTProfit = Constraint(                   rule=eTotalCost,         doc='total system cost [EUR]')

def eTotalRevenue(mBTC):
    return mBTC.vTotalRevenue == mBTC.vTotalProfit - mBTC.vTotalCost
mBTC.eTotalRevenue = Constraint(                   rule=eTotalRevenue,      doc='total system profit [EUR]')


def eObjFunction(mBTC):
    return mBTC.vTotalRevenue
mBTC.eObjFunction = Objective(rule=eObjFunction  , sense=maximize,          doc='total system revenue [EUR]')


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
print('Total system profit...................... ', round(mBTC.vTotalProfit()), '[EUR]')
print('Total system cost........................ ', round(mBTC.vTotalCost()), '[EUR]')
print('Total system revenue..................... ', round(mBTC.vTotalRevenue()), '[EUR]')


#%% Final Power Schedule

OfflineStates = pd.Series(data=[1.0 - mBTC.vOnlineStates[p,t,g]() for p,t,g in mBTC.ptg], index=pd.MultiIndex.from_tuples(mBTC.ptg))


#%% Output Results

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
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'],                    values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'PowerBuy {x} [MW]'     ).to_csv(_path+'/BTC_Result_PowerBuy_'     +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vPowerSell[p,t]()      for p,t   in mBTC.pt],             index=pd.MultiIndex.from_tuples(mBTC.pt))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1']                   , values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'PowerSell {x} [MW]'    ).to_csv(_path+'/BTC_Result_PowerSell_'    +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vPOutput[p,t,g]()      for p,t,g in mBTC.ptg],            index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'POutput {x} [MW]'      ).to_csv(_path+'/BTC_Result_PowerOutput_'    +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vQOutput[p,t,g]()      for p,t,g in mBTC.ptg],            index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QOutput {x} [MW]'      ).to_csv(_path+'/BTC_Result_HeatOutput_'     +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[mBTC.vQExcess[p,t,g]()      for p,t,g in mBTC.ptg],            index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='MW'  ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='MW'  ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'QExcess {x} [MW]'      ).to_csv(_path+'/BTC_Result_HeatExcess_'     +CaseName+'.csv', sep=',')

OutputResults = pd.Series(data=[(mBTC.pFuelCost[g] * mBTC.vFuel[p,t,g]()) for p,t,g in mBTC.ptg], index=pd.MultiIndex.from_tuples(mBTC.ptg))
OutputResults.to_frame(   name='EUR' ).reset_index().pivot_table( index=['level_0','level_1'], columns='level_2', values='EUR' ).rename_axis(['Period','LoadLevel'], axis=0).rename_axis([None], axis=1).rename(columns=lambda x: f'FuelCost {x} [EUR]'    ).to_csv(_path+'/BTC_Result_FuelCost_'       +CaseName+'.csv', sep=',')


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
dfQOutput           = pd.read_csv(_path + '/BTC_Result_HeatOutput_'     + CaseName + '.csv', index_col=[0, 1])
dfQExcess           = pd.read_csv(_path + '/BTC_Result_HeatExcess_'     + CaseName + '.csv', index_col=[0, 1])
dfFuelCost          = pd.read_csv(_path + '/BTC_Result_FuelCost_'       + CaseName + '.csv', index_col=[0, 1])

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
       dfDemand['PDemand'],
       dfQOutput,
       dfQExcess,
       dfDemand['QDemand']]

dfResult = pd.concat(dfs, join='outer', axis=1).to_csv(_path + '/BTC_Result_01_Summary_' + CaseName + '.csv', sep=',', header=True, index=True)


WritingResultsTime = time.time() - StartTime
StartTime          = time.time()
print('Writing output results................... ', round(WritingResultsTime), 's')
print('Total time............................... ', round(ReadingDataTime + SettingUpDataTime + SolvingTime + WritingResultsTime), 's')
print('\n #### Non-commercial use only #### \n')
