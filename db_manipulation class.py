import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

import pandas as pd
import sqlite3

# Load spreadsheet
xl = pd.ExcelFile('DCAwells_Solved/DCAwells_Solved/DCA_Well 1.xlsx')
#fileName = 'DCAwells_Solved/DCAwells_Solved/DCA_Well '+str(WellID) +'.xlsx'

#xl = pd.ExcelFile(fileName)
# Print the sheet names
print(xl.sheet_names)

# Load a sheet into a DataFrame by name: df1
df1 = xl.parse('DCARegression')

#create a database named "DCA.db" in the folder where this code is located
conn = sqlite3.connect("DCA.db")  #It will only connect to the DB if it already exists

#create data table to store summary info about each case/well
cur = conn.cursor()
#RUN THIS TO CREATE A NEW TABLE
cur.execute("CREATE TABLE DCAparams (wellID INTEGER,  qi REAL, Di REAL, b REAL)")
#cur.execute("CREATE TABLE DCAparams(ID INTEGER NOT NULL PRIMARY KEY, wellID INTEGER, qi REAL, Di REAl, b REAL)")
conn.commit()

dfLength = 24


wellID = 1

rateDF = pd.DataFrame({'wellID':wellID*np.ones(dfLength,dtype=int), 'time':range(1,dfLength+1),'rate':df1.iloc[8:32,1].values})
rateDF['Cum'] = rateDF['rate'].cumsum() #creates a column named 'Cum' in rateDF that is the cumulative oil production 

#insert data into the summary table
qi = df1.iloc[2,3]
Di = df1.iloc[3,3]
b  = df1.iloc[4,3]

cur.execute("INSERT INTO DCAparams VALUES ({},{},{},{})".format(wellID, qi, Di, b))
conn.commit()

t = np.arange(1,dfLength+1)
Di = Di/12   #convert to monthly

q = 30.4375*qi/((1 + b*Di*t)**(1/b))
Np = 30.4375*(qi/(Di*(1-b)))*(1-(1/(1+(b*Di*t))**((1-b)/b))) #30.4375 = 365.125/12 (avg # days in a month)
#Np is cum. prod.
error_q = rateDF['rate'].values - q
SSE_q = np.dot(error_q, error_q) #Sum of Square Error

errorNp = rateDF['Cum'].values - Np
SSE_Np = np.dot(errorNp,errorNp)


rateDF['q_model'] = q
rateDF['Cum_model'] = Np
# Use DataFrame's to_sql() function to put the dataframe into a database table called "Rates"
rateDF.to_sql("Rates", conn, if_exists="append", index = False)

# Read from Rates database table using the SQL SELECT statement
df1 = pd.read_sql_query("SELECT * FROM Rates;", conn)
df2 = pd.read_sql_query("SELECT * FROM DCAparams;", conn) #"SELECT *" is select all
    
conn.close()

#This connects to existing DCA.db because DCA.b already exists
conn = sqlite3.connect("DCA.db")

wellID = 7
df1 = pd.read_sql_query("SELECT * FROM Rates WHERE wellID = {};".format(wellID), conn)


#Custom Plot parameters
titleFontSize = 18
axisLabelFontSize = 15
axisNumFontSize = 13

currFig = plt.figure(figsize=(7,5), dpi=100)

# Add set of axes to figure
axes = currFig.add_axes([0.15, 0.15, 0.7, 0.7])# left, bottom, width, height (range 0 to 1)

# Plot on that set of axes
axes.plot(df1['time'], df1['Cum']/1000, color="red", ls='None', marker='o', markersize=5,label = 'well '+str(wellID) )
axes.plot(df1['time'], df1['Cum_model']/1000, color="red", lw=3, ls='-',label = 'well '+str(wellID) )
axes.legend(loc=4)
axes.set_title('Cumulative Production vs Time', fontsize=titleFontSize, fontweight='bold')
axes.set_xlabel('Time, Months', fontsize=axisLabelFontSize, fontweight='bold') # Notice the use of set_ to begin methods
axes.set_ylabel('Cumulative Production, Mbbls', fontsize=axisLabelFontSize, fontweight='bold')
axes.set_ylim([0, 1200])
axes.set_xlim([0, 25])
xticks = range(0,30,5) #np.linspace(0,4000,5)
axes.set_xticks(xticks)
axes.set_xticklabels(xticks, fontsize=axisNumFontSize); 

yticks = [0, 400, 800, 1200]
axes.set_yticks(yticks)
axes.set_yticklabels(yticks, fontsize=axisNumFontSize); 

currFig.savefig('well'+str(wellID)+'_Gp.png', dpi=600)



#def objFun(qi,Di,b,t):
#    q = qi/((1 + b*Di*t)**(1/b))
#
#    error_q = rateDF['rate'].values - 30.4375*q
#    SSE_q = np.dot(error_q,error_q)
#    
#    return SSE_q
#
#SSE_q = objFun(qi,Di,b,t)

#Code to delete some rows in a data table
#cur.execute("DELETE FROM DCAparams WHERE caseID = 1;")
#conn.commit()
