import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import  MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import warnings
warnings.filterwarnings("ignore")

frTRAIN = 0.8               # % size of training dataset
RNDN = 42                   # random state
nK = 12                     # initial guess: clusters
n_FEATURES = 25


PROCESSED_FILE_DATA = '../data/processed/processed_data.csv'
CATEGORICAL_COLUMNS = ['ext_allowed', 'parking_type', 'accessible', 'ev_charging', 'heated', 'rooftop_exposed',
       'has_monthly_rates', 'has_transient_rates','scan_to_pay_flag', 'iq_facility','covered_parking', 'inout_allowed','has_event_rates', 'on_site_staff','facility_type_under_el', 'facility_type_garage',
       'facility_type_valet_stand', 'facility_type_lot',
       'facility_type_personal_spot','average_star_rating']
OUTPUT_PATH = '../data/final/'


########################################################################################
'''
 Descripton: Read the data from path specified and store in pandas dataframe
 Output : dataframe loaded from the path (type: DataFrame)
'''
#########################################################################################
def load_processed_data():
    df  = pd.read_csv(PROCESSED_FILE_DATA)
    return df


########################################################################################
'''
 Descripton: As dataset has mixed type of data so,for Feature selection, PFA(Principal Feature Analysis)is used.
 Paper referred:  http://venom.cs.utsa.edu/dmz/techrep/2007/CS-TR-2007-011.pdf

 Class to perform feature selection with Principal Feature Analysis
'''
#########################################################################################
class PFA(object):
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features

    def fit(self, X):
        if not self.q:
            self.q = X.shape[1]

        sc = MinMaxScaler()
        
        X = sc.fit_transform(X)

        pca = PCA(n_components=self.q).fit(X)
        A_q = pca.components_.T

        kmeans = KMeans(n_clusters=self.n_features).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]


########################################################################################
'''
 Input: df: DataFrame with mixed type of variables (type: DataFrame), 
        n_features: No of features to be selected (type: int).
 Descripton: Read the data from path specified and store in pandas dataframe
 Output : List with selected features
'''
#########################################################################################
def get_PFA_Features(df,n_features):
    
    pfa = PFA(n_features)
    pfa.fit(df.iloc[:,1::])

    # To get the transformed matrix
    X = pfa.features_

    # To get the column indices of the kept features
    column_indices = pfa.indices_
    
    #column indices dictionary with keys as indices and values as column names
    column_dict = {i: col for i, col in enumerate(df.columns[1:])}
    
    #Columns selected by PFA
    column_names_to_keep = [column_dict[ind] for ind in column_indices]
    
    return column_names_to_keep

'''
Helper function: To assist in getting elbow point from sse scores 
'''
def knee_locator(sse):
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    return kl.elbow+1


########################################################################################
'''
 Input: df: DataFrame (type: DataFrame), 
        column_names_to_keep: Features selected in PFA and to be used in model training.
 Descripton: Read the data from path specified and store in pandas dataframe
 Output : DataFrame 
'''
#########################################################################################
def get_elbow_point(df, column_names_to_keep):
    # defining new dataframe with selected features
    df_new = df[column_names_to_keep]
    
    # converting categorical column names to indexes
    column_dict = {col: i for i, col in enumerate(column_names_to_keep)}
    
    # calculating inertia scores
    sse=[]
    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(df_new)
        sse.append(kmeans.inertia_)
       
    elbow_point = knee_locator(sse)
    return elbow_point


########################################################################################
'''
 Input: df: DataFrame (type: DataFrame) , 
 Descripton: Split the dataset in training and testing to see performance of the trained model
 Output : Training DataFrame, Testing DataFrame 
'''
#########################################################################################
def split_dataset(df):
    # split training vs test dataset
    df_train, df_test = train_test_split(df, train_size=frTRAIN, random_state=RNDN)
    return df_train, df_test


########################################################################################
'''
 Input: df: df_train: DataFrame to train the model (type: DataFrame),
            df_test:  DataFrame to test the model (type: DataFrame),
            elbow Point: number of clusters to be made (type: int)
        column_names_to_keep: Features selected in PFA and to be used in model training.
 Descripton: Read the data from path specified and store in pandas dataframe
 Output :  Trained K-Means model
'''
#########################################################################################
def train_model(df_train,df_test, elbow_point):
    model = KMeans(n_clusters=elbow_point, random_state=RNDN)
    res = model.fit_predict(df_train)

     # insert cluster labels as new column
    df_train.insert(0, "Cluster", res) 

    # training: get silhouette score
    sil_train = silhouette_score(df_train, res)
    print("training: silhouette score for", f'{elbow_point:.0f} clusters: {sil_train:.3f}')

    # testing: generate "Cluster" column based on optimal number of clusters
    pred = model.fit_predict(df_test)
    sil_test = silhouette_score(df_test, pred)
    print("testing: silhouette score for", \
    f'{elbow_point:.0f} clusters: {sil_test:.3f}. Variance vs training: {(sil_test / sil_train -1)*100:.1f}%')
    return model


#########################################################################################
'''
 Input: df: DataFrame (type: DataFrame) , 
        model: Trained K-Means model. 
 Descripton: Cluster for each facility is predicted and inserted in the DataFrame
 Output : DataFrame with Cluster added
'''
#########################################################################################  
def get_predictions(df,model):
    # get predictions
    predictions = model.fit_predict(df)
    # insert Cluster column in original dataframe
    df.insert(0, 'Cluster', predictions)
    return df


#########################################################################################
'''
 Input: df: DataFrame with Cluster inserted (type: DataFrame) ,      
 Descripton: Obtain insights about clusters, via aggregation function agg and the 
             column-specific lambda functions, identifies either the median or the
             mode for the records that belong to a given cluster.
             
 Output : DataFrame giving insights for each cluster, from different angles
'''
#########################################################################################  
def cluster_profile(df):
    dfc = df.groupby("Cluster").agg({ 
        "height_restriction": "median",
        "p4w_total_gross_revenue": "median",
        "p4w_sipp": "median",
        "user_issues_per_park": "median",
        "product_cpp": "median",
        'total_cpp': "median",
        "p4w_return_pct": "median",
        'customer_reviews_count': "median",
        "utilization": "median", 
        "searches": "median",
        "rental_count": "median",
        "p4w_new_drivers": "median",
        "num_stalls": "median",
        "tipp": "median",
        'customer_reviews_count': "median",
        "refund_pct": "median",
        "num_words_parking_instruction": "median",
        "ext_allowed": lambda x: x.value_counts().index[0],
        "parking_type": lambda x: x.value_counts().index[0],
        "accessible": lambda x: x.value_counts().index[0],
        "heated": lambda x: x.value_counts().index[0],
        "rooftop_exposed": lambda x: x.value_counts().index[0],
        "ev_charging": lambda x: x.value_counts().index[0],
        "has_monthly_rates": lambda x: x.value_counts().index[0],
        "has_transient_rates": lambda x: x.value_counts().index[0],
        "has_event_rates": lambda x:x.value_counts().index[0],
        "scan_to_pay_flag": lambda x: x.value_counts().index[0],
        "covered_parking": lambda x: x.value_counts().index[0],
        "inout_allowed": lambda x: x.value_counts().index[0],
        "has_event_rates": lambda x: x.value_counts().index[0],
        "on_site_staff": lambda x: x.value_counts().index[0],
        "accessible": lambda x: x.value_counts().index[0],    
        "iq_facility": lambda x: x.value_counts().index[0],    
        "facility_type_under_el": lambda x: x.value_counts().index[0],
        "facility_type_garage": lambda x: x.value_counts().index[0],
        "facility_type_valet_stand": lambda x: x.value_counts().index[0],
        "facility_type_lot": lambda x: x.value_counts().index[0],
        "facility_type_personal_spot": lambda x: x.value_counts().index[0],
        "average_star_rating":   lambda x: x.value_counts().index[0]  
             })
    
    return dfc


#########################################################################################
'''
 Input: df: DataFrame with Cluster inserted  (type: DataFrame) ,      
 Descripton: This method returns a sorted dataframe with ranks of clusters. This is done
             on bases of seven columns 'p4w_total_gross_revenue','rental_count',
             'p4w_new_drivers', 'p4w_return_pct', 'searches', 'utilization', 'num_stalls'.
             These features mainly give idea which facilities are most liked, used and
             generate high revenue to SpotHero. 
             
 Output : DataFrame with Score of each Cluster which are considered as ranks
'''
#########################################################################################
def get_ranks_of_cluster(df):
    cluster_profile_df = cluster_profile(df).T
    num_clusters = len(cluster_profile_df.columns)
    rank_dic = {i: 0 for i in range(num_clusters)}
    deciding_features = ['p4w_total_gross_revenue','rental_count','p4w_new_drivers', 
                         'p4w_return_pct', 'searches', 'utilization', 'num_stalls']
    for feature in deciding_features:
        cluster = cluster_profile_df.loc[feature].idxmax()
        rank_dic[cluster]+=1
    rank_df = pd.DataFrame.from_dict({'Cluster': rank_dic.keys(), 'Rank': rank_dic.values()})
    rank_df.sort_values(by = 'Rank', inplace=True, ascending =False)
    return rank_df


########################################################################################
'''
 Input: df (type: DataFrame), 
        ranked_clusters_df: DataFrame with ranks or score of the clusters (type: DataFrame)
 Descripton: Assign qualities to the facilities
'''
#########################################################################################
def set_qualities_to_facilites(df, ranked_clusters_df):
    ranked_dict = ranked_clusters_df.set_index('Cluster').to_dict()['Rank']
    df['Quality_Score'] = df['Cluster'].map(ranked_dict)
    qualities = df['Quality_Score'].unique()
    
    quality_dict = {max(qualities): 'Best', min(qualities): 'Poor'}
    
    if len(qualities) != len(quality_dict):
        for quality in qualities:
           
            if quality not in quality_dict.keys():
             
                quality_dict[quality] = 'Average'
    df['Quality'] = df['Quality_Score'].map(quality_dict)
    df.drop(columns= 'Quality_Score', inplace=True)

'''
Helper Function:  To get high and low quality clusters
Input: ranked cluster Dataframe (type: DataFrame)
Output: low_quality_clusters: list of low quality clusters (type: List).
        highest_quality_cluster: int of cluster of high quality (type: int). 
'''
def get_high_and_low_quality_clusters(ranked_clusters_df):
    low_quality_clusters = ranked_clusters_df[ranked_clusters_df.Rank.isin([0,1,2])]['Cluster'].to_list()
    highest_quality_cluster = ranked_clusters_df.reset_index().loc[0]['Cluster']
    return low_quality_clusters, highest_quality_cluster


'''Helper function: To compare values row by row
   Input: row1: (type: dict),
          row1: (type: dict)
   Output: columns that differ from rows (type: List)      
'''
def get_diff_cols_row_by_row(row1,row2):
    # row1 should be of lower quality whose features are to be drawn
    # row2 should be of higher quality
    different_col = [col for col in row1.keys() if (row1[col]!=row2[col])]
    return different_col


'''Helper function: To get features responsible for lower quality.
   Input: cluster_profile_df (type: DataFrame),
          low_quality_cluster (type: int)
          high quality cluster (type: int) 
   Output: dictionary with keys as features responsible for lower quality and values as the ideal one (type: dict)
'''
def get_columns_diff_intra_cluster(cluster_profile_df, highest_quality_cluster, low_quality_cluster ):
    # Categorical columns to be considered
    cat_col_to_consider = list(set(cluster_profile_df.T.reset_index()['index']).intersection(set(CATEGORICAL_COLUMNS)))
    
    
    # Dataframe for comparison of values of categorical columns
    cluster_comparison =cluster_profile_df.T[[low_quality_cluster,highest_quality_cluster]].T
    
    # Converting each column in cluster_comparison as dictionary
    row1 = cluster_comparison[cat_col_to_consider].loc[low_quality_cluster].to_dict()
    row2 = cluster_comparison[cat_col_to_consider].loc[highest_quality_cluster].to_dict()
    
    # Fetching columns that differ in values of rows
    different_col=get_diff_cols_row_by_row(row1, row2)
    
    # Fetch their corresponding calues and return it as output
    different_values ={col: row2[col] for col in different_col}
    return different_values

########################################################################################
'''
 Input: df (type: DataFrame), 
        cluster_profile_df: DataFrame giving insights for each cluster (type: DataFrame)
        ranked_clusters_df: DataFrame with ranks or score of the clusters (type: DataFrame)
 Descripton: Insert a column 'Features_affecting' to dataframe that consist of list
             of the features responsible for low quality.
 Output: DataFrame with three columns - Facility id, Quality, features responsible if
         Quality is low or average. (type: DataFrame)
'''
#########################################################################################
def fill_features_for_facilities_to_be_improvised(df,cluster_profile_df ,ranked_clusters_df):
    low_quality_clusters, highest_quality_cluster = get_high_and_low_quality_clusters(ranked_clusters_df)
    features = []
    for row in df.iterrows():
        cluster = row[1]['Cluster']
        if cluster in low_quality_clusters:
            low_quality_cluster = cluster

            # Dataframe for comparison of values of categorical columns
            cluster_comparison =cluster_profile_df.T[[low_quality_cluster,highest_quality_cluster]]
            different_col_values = get_columns_diff_intra_cluster(cluster_profile_df,highest_quality_cluster,
                                                                  low_quality_cluster)
            row1  = {col: row[1][col] for col in CATEGORICAL_COLUMNS }
            different_col = get_diff_cols_row_by_row(different_col_values, row1)
            if different_col == []:
            
                features.append(get_diff_cols_row_by_row(row1, cluster_comparison.T[CATEGORICAL_COLUMNS].loc[highest_quality_cluster].to_dict()))
            else:
                
                features.append(different_col)
        else:
            features.append('No Need')
    df['Features_affecting'] = features
    
    return df[['facility_id', 'Quality', 'Features_affecting']]


########################################################################################
'''
 Input: df (type: DataFrame), 
 Descripton: Writes file to yje specified output path
'''
#########################################################################################
def write_file_to_folder(df):
    df.to_csv(OUTPUT_PATH+'Final_Analysis.csv', index = False)



if __name__=='__main__':
    # Load processed data from local
    df = load_processed_data()
    print('......Processed Data Loaded')
        
    # Get Features with PFA
    column_names_to_keep = get_PFA_Features(df, n_FEATURES)
    print('......Features selected with PFA')
    # Split dataset
    df_train, df_test = split_dataset(df)
    print('......Data Split Done')
    # Get Elbow point
    elbow_point = get_elbow_point(df_train, column_names_to_keep)
    print('......Elbow Point is: '+str(elbow_point))
    # Get Trained Model
    model = train_model(df_train,df_test, elbow_point)
    print('......Model Trained')
    # Make Predictions
    df = get_predictions(df, model)
    print('......Predicted clusters')
    # Make Cluster Profile 
    cluster_profile_df = cluster_profile(df)
    print('......Cluster Profile Made')
    # Rank Clusters
    ranked_clusters_df = get_ranks_of_cluster(df)
    print('......Clusters Ranked')

    set_qualities_to_facilites(df,ranked_clusters_df)
    print('.......Qualities Assigned to the clusters')

    #Fill features to the facilities that need to be improvised
    df = fill_features_for_facilities_to_be_improvised(df, cluster_profile_df, ranked_clusters_df)
    print('.......Filled affecting features')

    # Write File to final data folder
    write_file_to_folder(df)
    print('.......File written in final data folder')





    



    












