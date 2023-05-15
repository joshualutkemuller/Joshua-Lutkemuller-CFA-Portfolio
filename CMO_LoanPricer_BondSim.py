# -*- coding: utf-8 -*-

#! python3

#this model was initially developed by @Maciej Sikora but was tweaked with the addition of a dynamic pull
#from Fred and dynamic assumptions like the speed of mean reversion, 10 year CMT std dev.,
#and the 10 year long run mean

#An additional Loan Pricing Model and Bond Simulation (assuming instantaneous shocks)
# was developed for Joshua Lutkemuller, CFA

import numpy_financial as npf 
import numpy as np
from math import atan, pi
import pandas as pd
import altair as alt
from fredapi import Fred
from datetime import datetime
from dateutil.relativedelta import relativedelta
import timeit
import math

import numpy as np
from scipy import optimize

class Bond:
    def __init__(self, face_value, coupon_rate, years_to_maturity, coupon_frequency):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.years_to_maturity = years_to_maturity
        self.coupon_frequency = coupon_frequency

    def price(self, yield_rate):
        coupon_payment = self.face_value * self.coupon_rate / self.coupon_frequency
        cash_flows = np.full(int(self.years_to_maturity * self.coupon_frequency), coupon_payment)
        cash_flows[-1] += self.face_value
        discount_factors = [1 / ((1 + yield_rate / self.coupon_frequency) ** i) for i in range(1, len(cash_flows) + 1)]
        return np.dot(cash_flows, discount_factors)

    def yield_to_maturity(self, price):
        def objective_function(ytm):
            return self.price(ytm) - price

        return optimize.newton(objective_function, self.coupon_rate)

def rate_shock_simulation(bond, rate_shock_range, time_horizon, reinvestment_rate):
    base_yield = bond.yield_to_maturity(bond.price(0))
    rate_scenarios = [(base_yield + rate_shock / 10000) for rate_shock in rate_shock_range]

    prices = [bond.price(y) for y in rate_scenarios]
    future_values = [(prices[i] * (1 + reinvestment_rate) ** time_horizon) for i in range(len(prices))]
    total_returns = [(future_values[i] - prices[0]) / prices[0] for i in range(len(prices))]

    return rate_scenarios, prices, future_values, total_returns



class LoanPricingModel:
    def __init__(self, principal, interest_rate, term, wavg_life, prepayment_rate,
                 expected_losses, dealer_flat, origination_cost, return_on_assets,
                 credit_reserve, servicing_cost, capital_held):
        self.principal = principal
        self.interest_rate = interest_rate
        self.term = term
        self.wavg_life = wavg_life
        self.prepayment_rate = prepayment_rate
        self.expected_losses = expected_losses
        self.dealer_flat = dealer_flat
        self.origination_cost = origination_cost
        self.return_on_assets = return_on_assets
        self.credit_reserve = credit_reserve
        self.servicing_cost = servicing_cost
        self.capital_held = capital_held
        
    def calculate_amortization_schedule(self):
        monthly_rate = self.interest_rate / 12
        num_payments = self.term * 12
        payment = (self.principal * monthly_rate) / (1 - (1 + monthly_rate)**(-num_payments))
        
        balance = self.principal
        interest_paid = 0
        principal_paid = 0
        prepayments = 0
        total_cashflows = 0
        
        for month in range(num_payments):
            interest = balance * monthly_rate
            principal = payment - interest
            balance -= principal
            interest_paid += interest
            principal_paid += principal
            
            # Apply prepayments
            if month < num_payments * self.prepayment_rate:
                prepayment_amount = balance * self.prepayment_rate
                balance -= prepayment_amount
                prepayments += prepayment_amount
            
            total_cashflows += payment
        
        return {
            'interest_paid': interest_paid,
            'principal_paid': principal_paid,
            'prepayments': prepayments,
            'total_cashflows': total_cashflows
        }
    
    def calculate_yield(self):
        amortization_schedule = self.calculate_amortization_schedule()
        total_cashflows = amortization_schedule['total_cashflows']
        total_cost = self.calculate_total_cost()
        yield_value = (total_cashflows - total_cost) / self.principal
        return yield_value
    
    def calculate_price(self):
        amortization_schedule = self.calculate_amortization_schedule()
        total_cashflows = amortization_schedule['total_cashflows']
        total_cost = self.calculate_total_cost()
        price = total_cost / (1 - np.exp(-self.wavg_life * self.interest_rate))
        return price
    
    def calculate_total_cost(self):
        total_cost = self.dealer_flat + self.origination_cost + self.calculate_return_on_assets() \
                     + self.calculate_credit_reserve() + self.calculate_servicing_cost() \
                     + self.capital_held * self.principal
        return total_cost
    
    def calculate_return_on_assets(self):
        return self.return_on_assets * self.principal
    
    def calculate_credit_reserve(self):
        return self.credit_reserve * self.expected_losses
    
    def calculate_servicing_cost(self):
        return self.servicing_cost * self.principal



#Mean reversion speed, K, is interpreted better with the concept of half-life, which is calculated
# as such, HL = ln(2)/K

#Select Half-Life variable
HL = 8
 
#Calculating reversion speed coefficient
var_k = math.log(2)/HL
print(var_k)

#functtion to generate the 10 year treasury curve from fred
#Please sign up for FRED and use your own API key
fred_api_key = '9f2245fa2cda35fec8864d56750b68f9'

#generate date 10 years ago
ten_years_ago = datetime.now() - relativedelta(years=10)

def Treasury_10yr_Curve():
    ''' The purpose of this code is to leverage the Fred API to pull the
    10 year CMT curve, filter only for dates that are within 10 years, and 
    generate key statistics as inputs to the CMO model'''

    fred = Fred(api_key=fred_api_key)
    data = fred.get_series('DGS10')
    df = pd.DataFrame({'Date':data.index,'10yr':data.values})
    df = df.copy().loc[df['Date'] >= ten_years_ago]
    
    #generate 10 year mean
    ten_year_mean_10yr = (df['10yr'].mean())/100

    #generate 10 year std. dev.
    ten_year_stddev_10yr = (df['10yr'].std())/100

    return df, ten_year_mean_10yr, ten_year_stddev_10yr

ten_year_treasury_df, ten_year_mean_10yr, ten_year_stddev_10yr = Treasury_10yr_Curve()

print(ten_year_mean_10yr,ten_year_stddev_10yr)

#function to generate path of interest rates with Cox-Ingersoll-Ross model 
#last -> last observation, starting point for the path
#a -> speed of adjustment to the mean
#b -> long-term mean
# sigma -> standard deviation 
#n -> number of interested rates to be generated by the function
def CIR(last, a, b, sigma, n):

    rates = np.array([last])
    for n in range(1,n):
        w = np.random.normal(0,1)
        dr = a * (b - rates[-1]) + sigma*w*(rates[-1]**(1/2))
        r = rates[-1] + dr
        rates = np.append(rates, r)
    return rates

#function to calculate value of Refinancing Incentive factor
# r -> current mortgage rate 
def RefinIncent(r, maxCPR, minCPR, WAC, slope):
    
    a = (minCPR+maxCPR)/2 
    b = (maxCPR-a)/(pi/2) 
    d = slope / b * 100
    c = -d * 0.02 #assuming 200 bps difference between C and R 
 
    return a + b*(atan(c+d*(WAC - r)))

#identifying month for Monthly Multiplier factor
def month(n):
    if n%12 != 0:
        m = (n%12) 
    else: 
        m = 12
    return m 

#funtion returning Single Month Mortality based on CPR calculated with Richard and Roll (1989) prepayment model 
def RichardRoll(r, maxCPR, minCPR, WAC, slope, balance, size, i):

    RI = RefinIncent(r, maxCPR, minCPR, WAC, slope)
    SM = min(i/30,1)
    
    #based on observations that prepayments tend to occur more frequently in certain months
    monthly_multiplier = np.array([0.94, 0.76, 0.74, 0.95, 0.98, 0.92, 0.98, 1.1, 1.18, 1.22, 1.23, 0.98])
    MM = monthly_multiplier[month(i)-1]
    BM = 0.3 + 0.7*(balance/size)
    
    cpr = RI * SM * MM * BM
    smm = 1 - (1-cpr)**(1/12)
    return smm 

#function to calculate monthly default rate according to SDA benchmark
def SDA(speed, month):
    if month <= 30:
        cdr = 0.0002 * (speed/100) * month
    elif month > 30 and month <= 60:
        cdr = 0.006 * (speed/100)
    elif month > 60 and month <= 120:
        cdr = 0.006 * (speed/100) - (month-60) * 0.000095 * (speed/100)
    else:
        cdr = 0.0003 * (speed/100)
        
    mdr = 1 - (1-cdr)**(1/12)
    return mdr

#function returning a dictionary of numpy arrays with cash flows generated by the pool
#size -> size of the pool (in millions)
#sda -> SDA (in %) 
#WAC -> Weighted Average Coupon
#netC -> net coupon (WAC - servicing fee)
#lag -> time to liquidation
#rr -> recovery rate (1 - loss severity)
#WAM -> Weighted Average Maturity 
def poolCashFlows(size, sda, WAM, WAC, netC, last, lag, rr):
    rates = CIR(last, var_k, ten_year_mean_10yr, ten_year_stddev_10yr, WAM) #calibrated values
    balance = size
    
    interestCF = np.array([])
    principalCF = np.array([]) 
    defaultED = np.array([])
    
    i = 0    
    while balance > 0:    
        interest = balance * (netC/12) 
        principal = -npf.pmt(netC/12, WAM-i, balance) - interest
        prepayment = (balance - principal) * RichardRoll(rates[i], 0.5, 0, WAC, 0.6, balance, size, i+1)
        default = (balance - (principal + prepayment)) * SDA(sda, i+1)
        
        if i > lag: 
            recovery = defaultED[i-lag-1] * rr
        else: 
            recovery = 0 
            
        if balance - principal - prepayment <0: 
            break
        elif i < lag:
            balance = balance - principal - prepayment - default 
        else:
            balance = balance - principal - prepayment - default - recovery 
        
        interestCF = np.append(interestCF, interest)
        principalCF = np.append(principalCF, principal + prepayment + recovery) 
        defaultED = np.append(defaultED, default)
        i += 1
    #cash flows in the final period 
    interest = balance * (netC/12)
    principal = balance
    recovery = defaultED[WAM-lag-2] * rr
    
    interestCF = np.append(interestCF, interest)
    principalCF = np.append(principalCF, principal + recovery)
    defaultED = np.append(defaultED, 0)
    totalCF = interestCF + principalCF 

    return {'interest':interestCF, 'principal':principalCF, 'defaults':defaultED, 'total': totalCF} 

#function to identify the month until which losses are covered by overcollateralization
def coveredLosses(a, oc, size):
    for j, cf in enumerate(np.cumsum(a)):
            if cf >= oc * size:
                break
    return j

#function that resets balance of each tranche    
def setBalance(tranches, size):
    for t, tranche in enumerate(tranches):
        tranche['balance'] = size * tranche['size']
        
#helper function to create additional arrays in dictionaries with tranche characteristics 
def addArraysToStore(tranches):
    for tranche in tranches:
        tranche['WAL'] = np.array([])
        tranche['yield'] = np.array([])
        tranche['avgWALs'] = np.array([])
        tranche['avgYields'] = np.array([])  

#distributing pools' cash flows to each tranche
#tranches -> list of dictionaries specifying tranches
#oc -> size of overcollateralization (in % of the size of the pool)
def waterfallCF(size, sda, WAM, WAC, netC, last, lag, rr, tranches, oc):

    setBalance(tranches, size)
    assets = poolCashFlows(size, sda, WAM, WAC, netC, last, lag, rr)
    
    for i, elem in enumerate(assets['interest']):
        
        avail_interest = assets['interest'][i]
        avail_principal = assets['principal'][i]
        avail_default = assets['defaults'][i]
        covered_period = coveredLosses(assets['defaults'], oc, size)
        
        for t ,tranche in enumerate(tranches):
            if t == len(tranches) - 1:
            #distributions for the Residual tranche
                if  i <= covered_period:
                    received_interest = avail_interest
                    received_principal = avail_principal
                        
                    tranche_cf = received_interest + received_principal
                    tranche['cash_flows'] = np.append(tranche['cash_flows'], tranche_cf)
                else:
                    avail_default = avail_default - avail_interest - avail_principal
                    for d in reversed(tranches):
                        decrease = min(d['balance'], avail_default)
                        d['balance'] = d['balance']  - decrease
                        avail_default = avail_default - decrease
                    
            else: 
            #disteributions for remaining tranches
                accrued_interest = tranche['balance'] * (tranche['coupon']/12)
                received_interest = min(avail_interest, accrued_interest)
                received_principal = min(avail_principal, tranche['balance'])
            
                tranche['balance'] = tranche['balance'] - received_principal
                
                avail_interest = avail_interest - received_interest
                avail_principal = avail_principal - received_principal 
                
                tranche_cf = received_interest + received_principal
                tranche['cash_flows'] = np.append(tranche['cash_flows'], tranche_cf)

#calculating WAL for each tranche
def WAL(tranche):
    wcf = []
    for i, elem in enumerate(tranche['cash_flows']):
        weighted_cf = (i+1) * elem
        wcf = np.append(wcf, weighted_cf)
    
    wal = np.sum(wcf) / np.sum(tranche['cash_flows'])
    return wal 

#calculating IRR (i.e. cash flow yield) for each tranche
def trancheIRR(size, tranche, tranches, oc, i):
    if i == len(tranches) - 1:
        a = np.array([-size*oc]) 
    else:
        a = np.array([-size*tranche['size']])
    irr = 12*npf.irr(np.append(a, tranche['cash_flows']))
    return irr

#function that resets cash flows distributed to each tranche before next simulation       
def zeroCashFlows(tranches):
    for tranche in tranches:
        tranche['cash_flows'] = np.array([])

#function that resets yields and WALs for each before next simulation     
def zeroMCKeys(tranches):
    for tranche in tranches:
         tranche['WAL'] = np.array([])
         tranche['yield'] = np.array([])
         
         
#function performing MC simulation and appending arrays in the tranches' dictionaries 
#n -> number of simulations
def MonteCarlo(size, sda, WAM, WAC, netC, last, lag, rr, tranches, oc, n):     
    for m in range(n):
        waterfallCF(size, sda, WAM, WAC, netC, last, lag, rr, tranches, oc)
        for i, tranche in enumerate(tranches):
            tranche['WAL'] = np.append(tranche['WAL'], WAL(tranche))
            tranche['yield'] = np.append(tranche['yield'],trancheIRR(size, tranche, tranches, oc, i))
        zeroCashFlows(tranches)
    for tranche in tranches:
        tranche['avgWALs'] = np.append(tranche['avgWALs'], np.mean(tranche['WAL']))
        tranche['avgYields'] = np.append(tranche['avgYields'], np.mean(tranche['yield']))
    zeroMCKeys(tranches)

#function to calcualte the data for the chart and store them in a csv file in the directory
def sourceData(sda_range, size, WAM, WAC, netC, last, lag, rr, tranches, oc, n): 

    addArraysToStore(tranches)
    for d in sda_range:
        MonteCarlo(size, d, WAM, WAC, netC, last, lag, rr, tranches, oc, n)
    
    df = pd.DataFrame({'x': [n/100 for n in sda_range], 
                        'A': tranches[0]['avgYields'],
                        'B': tranches[1]['avgYields'],
                        'C': tranches[2]['avgYields'],
                        'D': tranches[3]['avgYields'],
                        'Residual': tranches[4]['avgYields']
                        })
    
    df2 = pd.DataFrame({'x':[n/100 for n in sda_range],
                        'A': tranches[0]['avgWALs'],
                        'B': tranches[1]['avgWALs'],
                        'C': tranches[2]['avgWALs'],
                        'D': tranches[3]['avgWALs'],
                        'Residual': tranches[4]['avgWALs']
                        })
    
    source = pd.melt(df, id_vars =['x'], value_vars =['A', 'B', 'C', 'D', 'Residual'], var_name='variable')
    source2 = pd.melt(df2, id_vars =['x'], value_vars =['A', 'B', 'C', 'D', 'Residual'], var_name='variable')
    
    source.to_csv('source.csv', index=False)
    source2.to_csv('source2.csv',index=False)
    yieldSensitivityChart(source, sda_range)
    WALSensitivityChart(source2,sda_range)

#'Sensitivity of CMO Yield' chart
#source -> needs to be pandas Data Frame
def yieldSensitivityChart(source, sda_range):

    chart = alt.Chart(source).mark_line(point=True, size=5).encode(
    x= alt.X('x', axis=alt.Axis(values=[n/100 for n in sda_range], format='%', title='SDA', labelSeparation=10, labelFlush=False)),
    y= alt.Y('value', axis=alt.Axis(tickCount=10, format='%', title='Yield')),
    color= alt.Color('variable', scale=alt.Scale(scheme = 'blues'), legend=alt.Legend(labelFont='Open Sans',
                                                                                      labelFontSize=14,
                                                                                      titleFont='Open Sans',
                                                                                      titleFontSize=14,
                                                                                      titleFontWeight='normal',
                                                                                      title="Class:")
                    ) 
    ).properties(title="Sensitivity of CMO Yield", width=600, height=350)
    
    chart.configure_title(
        fontSize=32,
        font='Lato Light',
        align='center',
        color='black',
        fontWeight=100
    ).configure_axis(
        titleFont='Lato Light',
        titleFontSize=25,
        titleFontWeight=200,
        labelFont='Open Sans',
        labelFontSize=14,
        labelPadding=10,
        gridColor='#e2e2e2'
    ).configure_point(
        size=60
    ).display()

def WALSensitivityChart(source, sda_range):

    chart = alt.Chart(source).mark_line(point=True, size=5).encode(
    x= alt.X('x', axis=alt.Axis(values=[n/100 for n in sda_range], format='%', title='SDA', labelSeparation=10, labelFlush=False)),
    y= alt.Y('value', axis=alt.Axis(tickCount=10, title='Weighted Average Life (Months)')),
    color= alt.Color('variable', scale=alt.Scale(scheme = 'blues'), legend=alt.Legend(labelFont='Open Sans',
                                                                                      labelFontSize=14,
                                                                                      titleFont='Open Sans',
                                                                                      titleFontSize=14,
                                                                                      titleFontWeight='normal',
                                                                                      title="Class:")
                    ) 
    ).properties(title="Sensitivity of CMO WALs", width=600, height=350)
    
    chart.configure_title(
        fontSize=32,
        font='Lato Light',
        align='center',
        color='black',
        fontWeight=100
    ).configure_axis(
        titleFont='Lato Light',
        titleFontSize=25,
        titleFontWeight=200,
        labelFont='Open Sans',
        labelFontSize=14,
        labelPadding=10,
        gridColor='#e2e2e2'
    ).configure_point(
        size=60
    ).display()

###------------------------------
#Running CMO Model 
#defining tranches as a list of dictionaries 
#tranches must be in order of seniority
tranches = [
    {'size':0.60, 'coupon':0.010, 'balance':0, 'cash_flows': np.array([])},
    {'size':0.15, 'coupon':0.020, 'balance':0, 'cash_flows': np.array([])},
    {'size':0.15, 'coupon':0.035, 'balance':0, 'cash_flows': np.array([])},
    {'size':0.03, 'coupon':0.055, 'balance':0, 'cash_flows': np.array([])},
    {'size':0.00, 'coupon':0.000, 'balance':0, 'cash_flows': np.array([])} #Residual tranche
           ]

#specifying SDA levels for sensitivity analysis
#intervals of 50 starting at 50 and going up till 425 (excludes 425)
sda_range = list(range(50,425,25))

#applying sample values from the example
sourceData(sda_range, 300_000_000, 357, 0.035, 0.03, 0.0272, 14, 0.8, tranches, 0.07, 1)

###-------------------------------
#Loan Pricing Model
#Feel free to use and tweak pricing model that takes into account normal FTP assumptions
# pricing_model = LoanPricingModel(
#     principal=1000000,
#     interest_rate=0.05,
#     term=5,
#     wavg_life=3,
#     prepayment_rate=0.05,
#     expected_losses=0.01,
#     dealer_flat=1000,
#     origination_cost=0.01,


###--------------
#Bond Model
# # Example bond
# face_value = 1000
# coupon_rate = 0.05
# years_to_maturity = 5
# coupon_frequency = 2

# bond = Bond(face_value, coupon_rate, years_to_maturity, coupon_frequency)

# # Rate shock range
# rate_shock_range = list(range(-300, 301, 1))

# # Input time horizon and reinvestment rate
# time_horizon = 5  # in years
# reinvestment_rate = 0.04  # as a decimal

# # Calculate rate shock scenarios
# rate_scenarios, prices, future_values, total_returns = rate_shock_simulation(bond, rate_shock_range, time_horizon, reinvestment_rate)

# # Print results
# for i, rate_shock in enumerate(rate_shock_range):
#     print(f"Rate shock: {rate_shock} bps | Yield: {rate_scenarios[i] * 100:.2f}% | Price: ${prices[i]:.2f} | Future value: ${future_values[i]:.2f} | Total return: {total_returns[i] * 100:.2f}%")
