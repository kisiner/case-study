import os 
import math
import pandas as pd
import random      
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


periods_cds = ['2 years', '5 years', '10 years']
columns_inf = ['Reference','Spread','Change','Change3m','Low','High','Avg','Column2','Column3']

bonds = pd.read_excel('./bonds.xlsx', usecols = 'A:E,H:M',index_col=None)
cds = pd.read_excel('./cds_by_countries.xlsx' ,sheet_name = periods_cds, header=None, names = columns_inf, skiprows=5, usecols = 'B,D:G,J:M')

###################################Data cleaning###################################

bonds['Today'] = None
bonds['Today'] = pd.to_datetime('2023-11-24')

bonds_cols = ['Cpn','Maturity','Yld to Mty (Ask)','Yld to Mty (Bid)']


# Replace specific values with NaN
bonds.replace(['#N/A Invalid Security', '#N/A Field Not Applicable'], np.nan, inplace=True)

# Drop rows with NaN values
bonds.dropna(subset=bonds_cols, inplace=True)


#bonds.loc[:,('Maturity')] = pd.to_datetime(bonds.loc[:,('Maturity')])
bonds['Maturity'] = pd.to_datetime(bonds['Maturity'])
#bonds['Maturity'].astype(np.datetime64)
#Droping the expired bonds
bonds = bonds[bonds['Today']<bonds['Maturity']]


# Reset the index if needed
bonds.reset_index(drop=True, inplace=True)



#Converting rates to percentage points
bonds['Cpn'] = bonds['Cpn']/100
bonds['Yld to Mty (Ask)'] = bonds['Yld to Mty (Ask)']/100
bonds['Yld to Mty (Bid)'] = bonds['Yld to Mty (Bid)']/100


#CDS data cleaning
for p in periods_cds:
    cds[p]['Asterix'] = cds[p]['Spread'].str[-1] == '*'
    
    cds[p]['Spread'] = cds[p].loc[:, 'Spread'].str.rstrip('*')
    cds[p]['Spread'] = cds[p]['Spread'].astype(float)

    

##################################Data generation##################################

################################# Bonds Data


bonds['MtM'] = None
bonds['YtM'] = None


#bonds['Period'] = None


# MtM and YtM clean integer

bonds['MtM'] = (bonds['Maturity'] -bonds['Today'])/np.timedelta64(1, 'M').astype(int)
bonds['MtM'] = (pd.to_numeric(bonds['MtM'], downcast='integer') / (10 ** 9 * 60 * 60 * 24 * 30)).astype(int)


bonds['YtM'] = (bonds['Maturity'] -bonds['Today'])/np.timedelta64(1, 'Y').astype(int)
bonds['YtM'] = (pd.to_numeric(bonds['YtM'], downcast='integer') / (10 ** 9 * 60 * 60 * 24 * 30 * 12 )).astype(int)


### Assigns random entities to reference entity column of CDS data 
#(creates artificial pispricing on CDS bond pairings only 2condition United states applies)


bonds['Entity'] = None

entities = list(cds['2 years']['Reference'])



bonds['GradeHigh'] = None

grade_high = ['A', 'A+', 'A-', 'AA', 'AA+', 'AA-', 'AAA'] #Low credit risk


for i in range(len(bonds)):    
    bonds.loc[i,('GradeHigh')] = bonds.loc[i,('BBG Composite')] in grade_high
    
    if'United States' in bonds.loc[i,('Issuer Name')]:
        bonds.loc[i,('Entity')] = 'United States'
    elif 'Federal' in bonds.loc[i,('Issuer Name')]:
        bonds.loc[i,('Entity')] = 'United States'
    else:
        bonds.loc[i,('Entity')] = random.choice(entities)
        
##################################Portfolio##################################
    

#chooses highestcoupon paying  grade and junk bonds 
#started at Jan 13 15.30  end at 18.00


class portfolio:
    def __init__(self,bonds,period,num_g,num_j):
        
        self.bonds = bonds[bonds['Currency'] != 'EUR']
        self.period = period
        self.num_j = num_j
        self.num_g = num_g
        self.ratio = self.num_g/(self.num_g+self.num_j)
        
        
        
        self.lot_size_s = 10000000/(self.num_g + self.num_j)
        
        self.cpn_return_s = 0
        
        self.grade = self.bonds['GradeHigh']
        self.entity = self.bonds['Entity']
        self.m_maturity = self.bonds['MtM']
        self.y_maturity = self.bonds['YtM']
        self.currency = self.bonds['Currency']
        
    
##### Assign portfolio mix period static
        self.conditions_grade =  self.grade | (( self.entity == 'United States') & ( self.m_maturity > 12) & self.grade )
        self.conditions_junk = ~self.grade.astype(bool) & (self.m_maturity >12 ) 
        
        self.junk = self.bonds[self.conditions_junk].sort_values(by = ['Cpn'], ascending = False)[:self.num_j]
        self.grade = self.bonds[self.conditions_grade].sort_values(by=['Cpn'], ascending = False)[:self.num_g]
        
        self.static = pd.concat([self.grade,self.junk])
        
        #self.a = self.porfolio_junk['Cpn']
        ######## calculate return up to self.years
         
        
        for i in range(len(self.static)):
            self.cpn_return_s += self.lot_size_s * float(self.static.iloc[[i]]['Cpn'])
        
        
##### Assign portfolio period t dynamic  
# ladders the bonds wrt maturity
#bond allocation


       
    def ladder(self):
        
        self.dynamic = pd.DataFrame(columns= self.bonds.columns.tolist())
        
        for year in range(1,self.period+1):
                        
            condition = self.conditions_grade & (self.y_maturity  == year)            
                        
            self.dynamic = self.dynamic.append(self.bonds[condition].sort_values(by = ['Cpn'], ascending = False)[:self.num_g])
            
            self.dynamic = self.dynamic.append(self.bonds[condition].sort_values(by = ['Cpn'], ascending = False)[:self.num_j])
            
        

        self.cpn_return_d = 0
        self.lot_size_d = 10000000/(self.num_g + self.num_j)*self.period
    
        for i in range(len(self.dynamic)):
            self.cpn_return_d += self.lot_size_d * float(self.dynamic.iloc[[i]]['Cpn'])
        
        return self.dynamic




##################################Search##################################


#track1.portfolio_bonds.groupby(['Grade','Series']).count().sort_values('Grade',ascending=False)

search_grid = [1,3,5,10,25,100]

portfolio_size = {'Grade': list(),'Junk': list(), 'CouponReturn':list(), 'Ratio':list()}


for g in search_grid:
    for j in search_grid:
        track = portfolio(bonds,'2 Years', num_g = g, num_j = j)
        
        portfolio_size['Grade'].append(track.num_g) 
        portfolio_size['Junk'].append(track.num_j) 
        portfolio_size['CouponReturn'].append(track.cpn_return_s) 
        portfolio_size['Ratio'].append(track.ratio) 

heat =  pd.DataFrame.from_dict(portfolio_size)


##################################Analysis##################################

   # at this point I realized how to connect with CDS data (calculate CDS returns and compare it with government bonds CDSpayments = nbondyield - nriskfree)
    #Graph order of 10
##### Search space (Convex)

track1 = portfolio(bonds,'2 Years', num_g = 50, num_j = 100)

print('Ratio: {:.2f} LotSize: {:.2f} dollars  CouponReturnofYear: {:.2f} dollars'.format(track1.ratio,track1.lot_size_s,  track1.cpn_return_s))


track3 = portfolio(bonds,'2 Years', num_g = 75, num_j = 75)

print('Ratio: {:.2f} LotSize: {:.2f} dollars  CouponReturnofYear: {:.2f} dollars'.format(track3.ratio, track3.lot_size_s,  track3.cpn_return_s))



track2 = portfolio(bonds,'2 Years', num_g = 100, num_j = 50)

print('Ratio: {:.2f} LotSize: {:.2f} dollars  CouponReturnofYear: {:.2f} dollars'.format(track2.ratio,track2.lot_size_s,  track2.cpn_return_s))


##################################Visuals##################################

plt.scatter( heat['Ratio'],heat['CouponReturn'])
plt.xlabel('Ratio')
plt.ylabel('Return')
plt.show()



heat_cpn = heat.pivot(index="Grade", columns="Junk", values="CouponReturn")
sns.heatmap(heat_cpn)
plt.scatter(heat['Ratio'],heat['CouponReturn'])

plt.xlabel('Ratio')
