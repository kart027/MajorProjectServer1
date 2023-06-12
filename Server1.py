from flask import Flask,jsonify,request
from flask_restful import Api,Resource
import numpy as nm    
from pymongo import MongoClient
  
import pandas as pd
import pandas as pd
import pandas as pdd

import plotly.express as px

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
#finding optimal number of clusters using the elbow method  
from sklearn.cluster import KMeans  

import matplotlib
import json
matplotlib.use('SVG')





app = Flask(__name__)
api = Api(app)
# importing libraries    

client = MongoClient('mongodb+srv://kartikgamot2003:0xyBuUuQcaO4xg0G@cluster0.nh9tvpe.mongodb.net/')

db = client.flask_db
todos = db.Database

with open("dataset.json") as f:
    data = json.load(f)
    
todos.insert_many(data)
# todos.delete_many({})


def getMappingDictionary(unique_data):
    mappingDictionary = {}
    for i in range(len(unique_data)):
        mappingDictionary[unique_data[i]] = i
    return mappingDictionary


@app.route('/addEntry',methods=['POST'])
def okk():
     content = request.json
     print(content)
     age = content['age']
     gender =content['gender']
     intrest =content['interest']
     relatioship = content['relationship']
     occasion = content['occasion']
     budget = content['budget']
     rating =  content['rating']
     Link =  content['Link']
     Image_Link =  content['ImageLink']
     MaxBudget =  content['budget']
     GiftName =  content['Gift']
     
     user_input = {
        'Age':[age],
        'Gender':[gender],
        'Relationship':[relatioship],
        'Occasion':[occasion],
        'Budget':[budget],
        'MaxBudget':[MaxBudget],
        'Gift':[GiftName],
        'Rating':[rating],
        'Link':[Link],
        'Image Link':[Image_Link],
        'Interest':[intrest]
     }
     todos.insert_one(user_input)
     todos.delete_many({"Gift":"https//amazon.com"})
     
     
     return "ok"
     
     

@app.route('/getGift',methods=['POST'])
def ok():
    
     content = request.json
     age = content['age']
     gender =content['gender']
     intrest =content['interest']
     relatioship = content['relationship']
     occasion = content['occasion']
     budget = content['budget']
     
    
     # todos.delete_many({})
    

     
     user_input = {
        'Age':age,
        'Gender':gender,
        'Relationship':relatioship,
        'Occasion':occasion,
        'Budget':budget,
        'MaxBudget':500,
        'Gift':"https//amazon.com",
        'Rating':1,
        'Link':"https//amazon.com",
        'Image Link':"https//amazon.com",
        'Interest':intrest

     }
     todos.insert_one(user_input)
     k = todos.find({},{"_id":0})
     dataset = pd.DataFrame(list(k))
     # print(dataset)
          
     
#      # print(user_input) 
#      dataset = pd.read_csv('dataset.csv') 
#      neww = pdd.DataFrame(user_input);
#      neww.to_csv('dataset.csv', mode='a', index=False, header=False)
#      p = pd.read_json('dataset.json')
#      p.to_csv()
#      print(p)
#     # dataset.head(150)



     df = pd.DataFrame(dataset)
     newdf = df.copy()
    


# function to create a dictionary of unique values of column mapped to numerical values
     def getMappingDictionary(unique_data):
       mappingDictionary = {}
       for i in range(len(unique_data)):
           mappingDictionary[unique_data[i]] = i
       return mappingDictionary





     def getUniqueDataList(columnName):
      return list(set(dataset[columnName]))





     def modifyDatasetColumn(columnName, mapping_dict):
      dataset[columnName] = dataset[columnName].map(mapping_dict)





     def modifyDataset(columnName):
       unique_columnValues = getUniqueDataList(columnName)
     #   print(unique_columnValues)
       mapping_dict = getMappingDictionary(unique_columnValues)
     #   print(mapping_dict)
       modifyDatasetColumn(columnName,mapping_dict)



     modifyDataset('Gender')
    #  print(dataset.head())

   



     modifyDataset('Interest')
# dataset.head()





# extract headers from the database
     dataframe_headers = dataset.columns.values






# drop unnecessary columns from dataframe
     df.drop(['Relationship','Occasion','Budget','MaxBudget','Gift','Rating','Link','Image Link'],axis=1, inplace=True)
     

     # print(df)



     new_headers = df.columns.values

     # print(df)



     
     model = KMeans()
     visualizer = KElbowVisualizer(model, k=(1, 10)) 

# Fit the data to the visualizer
     visualizer.fit(df)

# # Display the elbow plot
# visualizer.show()
     k= visualizer.elbow_value_

     # print(k)

    

#training the K-means model on a dataset 
     kmeans = KMeans(n_clusters = k, init='k-means++', random_state= 42)  
     customer_segments = kmeans.fit_predict(df)
    
# [df['Age'].iloc[-1]]





# Create a new customer and predict the segment it belongs to
     input_customer_data = {
       'Age': [df['Age'].iloc[-1]], 
       'Gender':[df['Gender'].iloc[-1]], 
       'Interest': [df['Interest'].iloc[-1]],
    }

     new_customer = pd.DataFrame(input_customer_data)
     new_customer_segment = kmeans.predict(new_customer[new_headers])
     new_customer_segment[0]


     labels = kmeans.labels_
     centroids = kmeans.cluster_centers_
     df2 = pd.DataFrame()
     df2.index.name = 'id'
     df2['Relationship']= newdf.iloc[:,2]
     df2['Occasion']= newdf.iloc[:,3]
     df2['Budget']= newdf.iloc[:,5]
     df2['Interest']= newdf.iloc[:,10]
     df2['Gift']= newdf.iloc[:,6]
     df2['Rating']= newdf.iloc[:,7]
     df2['Link'] = newdf.iloc[:,8]
     df2['Image Link'] = newdf.iloc[:,9]
     df2['Cluster']= customer_segments
    




     Newdata = pd.DataFrame()
     Newdata=df2[df2['Cluster'] == new_customer_segment[0]]
# Newdata 



 

     Newdata.index.name = '_id'
# dataset['Relationship'].iloc[-1]
     dataset2 = pd.read_csv('dataset.csv') 





     recommended_products = Newdata[Newdata['Rating'] >= 2.5]
     temp = recommended_products
     print("occasion data",  occasion)
     print("interest data",  intrest)
     if( len(temp)>5): 
          temp = recommended_products[recommended_products['Interest'] ==  intrest]
     if(len(temp)>5):
          recommended_products=temp     
     if( len(recommended_products)>5): 
          temp = recommended_products[recommended_products['Occasion'] ==  occasion]
     if(len(temp)>5):
          recommended_products=temp
     if(len(recommended_products)>5): 
          temp = recommended_products[recommended_products['Relationship'] == relatioship]
     if(len(temp)>5):
          recommended_products=temp

     # temp = recommended_products[recommended_products['Budget'] >= 0]
     # if(len(temp)>=5):
     #           recommended_products=temp
     # temp = recommended_products[recommended_products['MaxBudget'] <= (50000+5000)]
     # if(len(temp)>=5):
     #           recommended_products=temp
     
     recommended_products = recommended_products.sort_values(by=['Rating'],ascending=False)

     print(recommended_products['Gift'])
     final_data = []
     # print(recommended_products['Gift'])
     
     def checkDuplicate(row):
          for i in final_data:
             if i['Gift']==row['Gift']:
                 return False
          return True


     i=8
     for index, row in recommended_products.iterrows():
      data = {}
      if i>0 and checkDuplicate(row):
        data['Gift'] = row['Gift']
        data['ImageLink'] = row['Image Link']
        data['Link'] = row['Link'] 
        data['Rating'] = row['Rating'] 
        final_data.append(data)
        i=i-1
     return jsonify(final_data);

if __name__ == "__main__":
    app.run(debug=True)