import re, time, urllib.request
import os
import numpy as np
from sklearn.cluster import KMeans
import pickle
import math
import os
from datetime import datetime

'''
THIS PYTHON SCRIPT WILL DOWNLOAD THE DATASET
AND FORMAT IT IN A PROPER METHOD WITH FOR USE
WITH THE NEURAL NETWORK.
WILL CREATE A FILE CALLED Kingston_Police_Formatted.csv
WHICH IS THE FORMATTED POLICE DATA.
RUN THIS FIRST
'''



#remove old dataset
if (os.path.isfile('Kingston_Police.csv')):
    os.remove('Kingston_Police.csv')

#calls and retrieves the data set
urllib.request.urlretrieve("https://moto.data.socrata.com/api/views/fjwd-syvh/rows.csv?accessType=DOWNLOAD","Kingston_Police.csv")

'''
Function that sorts data into proper
human readable format using an alphanumerical strategy
'''
def sort_nicely( l ):
    #function that converts string to int if possible
    convert = lambda text: int(text) if text.isdigit() else text

    #sort data by highest int value
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

'''
initializing all necessary variables to use
'''
incident_date_time=[]
hour_of_year = []
sin_cos_time = []
incident_type = []
incidenttype_set = set()
incident_lat=[]
incident_long=[]
datetimearr = []
Hour_of_Day_Cos = []
Hour_of_Day_Sin = []
Day_of_Year = []
Day_of_Week = []
Week_of_Year = []
Year_Arr=[]
loc_cluster = []
'''
Loads csv columns into arrays
Filters through and prepares data
for exporting in formated manner
'''
i = 0;
for lines in open("Kingston_Police.csv","r"):
    if i == 0:
        labels=lines
    else:
        data = lines.split(',') #split row of csv
        date_time = datetime.strptime(data[2], '%m/%d/%Y %I:%M:%S %p') #convert entry 3 to datetime object
        hour_of_year = date_time.timetuple().tm_yday*24 + date_time.hour
        datetimearr+=[date_time]
        #convert incident_date_time into sin(2*pi*(time/maxtime)) and cos(2*pi*(time/maxtime))
        sin_cos_time += [[math.sin(2*math.pi*(hour_of_year/(365*24))), math.cos(2*math.pi*(hour_of_year/(365*24)))]];

        incident_date_time += [int(time.mktime(date_time.timetuple()))] #convert datetime to integer? idk
        incidenttype_set.add(data[3].lower())
        incident_type += [data[3].lower()]
        incident_lat += [float(data[12])]
        incident_long += [float(data[13])]
        loc_cluster += [[float(data[12]),float(data[13])]]
    i += 1
'''
To use in order to normalize the latitude and longitude
'''
latmax = max(incident_lat)
latmin = min(incident_lat)
longmin = min(incident_long)
longmax = max(incident_long)

#normalize all latitude/longitude to values between 0-1
normalized_lat = []
normalized_long = []
for k in incident_lat:
    normalized_lat += [(k-latmin)/(latmax-latmin)]

for l in incident_long:
    normalized_long +=[(l-longmin)/(longmax-longmin)]

#convert crime types to vectors of len(5)
incident_type_unique = sort_nicely(list(incidenttype_set))

'''
import kmeans clusters. These are generated by running cluster.py
'''
c1 = pickle.load(open("clust1.p","rb"))
c2 = pickle.load(open("clust2.p","rb"))
c3 = pickle.load(open("clust3.p","rb"))
c4 = pickle.load(open("clust4.p","rb"))
c5 = pickle.load(open("clust5.p","rb"))
c6 = pickle.load(open("clust6.p","rb"))



'''
Turns a string of incident_type into a one hot array
and predict cluster belonging
'''


for x in incident_type:
    val = incident_type_unique.index(x)
    i= incident_type.index(x)
    if (val==0):
        incident_type[i]=[1,0,0,0,0,0]
        loc_cluster[i] = c1.predict(loc_cluster[i])
    elif (val ==1 or val==2):
        incident_type[i]=[0,1,0,0,0,0]
        loc_cluster[i] = c2.predict(loc_cluster[i])
    elif (val==3):
        incident_type[i]=[0,0,1,0,0,0]
        loc_cluster[i] = c3.predict(loc_cluster[i])
    elif (val==4 or val==5):
        incident_type[i]=[0,0,0,1,0,0]
        loc_cluster[i] = c4.predict(loc_cluster[i])
    elif (val==6):
        incident_type[i]=[0,0,0,0,1,0]
        loc_cluster[i] = c5.predict(loc_cluster[i])
    else:
        incident_type[i]=[0,0,0,0,0,1]
        loc_cluster[i] = c6.predict(loc_cluster[i])

'''
Formatting date time into multiple usable formats
for testing
'''
for x in datetimearr:
    MinofDay = (x.timetuple().tm_hour*60)+ x.timetuple().tm_min
    Hour_of_Day_Cos+=[math.cos(2*math.pi*(MinofDay/(24)))]
    Hour_of_Day_Sin+=[math.sin(2*math.pi*(MinofDay/(24)))]
    Day_of_Year+=[x.timetuple().tm_yday]
    Day_of_Week+=[x.timetuple().tm_wday]
    Week_of_Year+=[math.floor(x.timetuple().tm_yday/7)]
    Year_Arr+=[x.timetuple().tm_year]


final_list = []
'''
Create final list to export
'''
for x in range(len(incident_type)):
    final_list+=[[incident_type[x][0],incident_type[x][1],incident_type[x][2],incident_type[x][3],incident_type[x][4],incident_type[x][5],incident_date_time[x],incident_lat[x],incident_long[x],normalized_lat[x],normalized_long[x],sin_cos_time[x][0],sin_cos_time[x][1],Hour_of_Day_Cos[x],Hour_of_Day_Sin[x],Day_of_Year[x],Day_of_Week[x],Week_of_Year[x],Year_Arr[x],loc_cluster[x]]]

final_arr = np.array(final_list)
'''
Sort array chronologically using time
in case certain elements are not chronological
'''
final_arr = final_arr[final_arr[:,6].argsort()]

if (os.path.isfile('Kingston_Police_Formatted.csv')):
    os.remove('Kingston_Police_Formatted.csv')
#write formatted data to Kingston_Police_Formatted.csv
with open('Kingston_Police_Formatted.csv','a') as out:
    for x in range(len(incident_type)):
        out.write(str(final_arr[x][0])+","+str(final_arr[x][1])+","+str(final_arr[x][2])+","+str(final_arr[x][3])+","+str(final_arr[x][4])+","+str(final_arr[x][5])+","+str(final_arr[x][6])+","+str(final_arr[x][7])+","+str(final_arr[x][8])+","+str(final_arr[x][9])+","+str(final_arr[x][10])+","+str(final_arr[x][11])+","+str(final_arr[x][12])+","+str(final_arr[x][13])+","+str(final_arr[x][14])+","+str(final_arr[x][15])+","+str(final_arr[x][16])+","+str(final_arr[x][17])+","+str(final_arr[x][18])+","+str(final_arr[x][19])+"\n")
'''
0:1st incident type
1:2nd incident type
2:3rd incident type
3:4th incident type
4:5th incident type
5:6th incident type
6:time in seconds since 1970
7:incident latitude
8:incident longitude
9:normalized lat
10:normalized long
11:sin cos hour of year (sin)
12:sin cos hour of year (cos)
13:Hour of Day Cos
14:Hour of Day sin
15:Day of Year
16:Day of Week
17:Week of Year
18:Year
19:Location cluster index
'''
