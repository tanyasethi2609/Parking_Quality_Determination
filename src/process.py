# Load libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Raw data path
RAW_DATA_PATH = '../data/raw/Parking_Facility_Data.csv'
OUTPUT_PATH = '../data/processed/'
REDUNDANT_COLUMN_LIST = ['city','longitude','latitude', 'lot_fulls_per_park','sipp','num_rentals_refunded', 'p4w_repeat_drivers',
                  'count_first_rentals', 'count_repeat', 'p4w_total_gross_revenue_no_event', 'utilization_weekend_night','utilization_weekday_day', 'latitude','longitude']
COLUMNS_WITH_PERCENT_SIGN = ['refund_pct','p4w_return_pct']

CATEGORICAL_COLUMNS = ['ext_allowed', 'parking_type', 'accessible', 'ev_charging', 'heated', 'rooftop_exposed',
       'has_monthly_rates', 'has_transient_rates','scan_to_pay_flag', 'iq_facility','covered_parking', 'inout_allowed','has_event_rates', 'on_site_staff','facility_type_under_el', 'facility_type_garage',
       'facility_type_valet_stand', 'facility_type_lot',
       'facility_type_personal_spot','average_star_rating']
DEMOCRATIC_COLUMNS = ['facility_id', 'reporting_neighborhood']



########################################################################################
'''
 Descripton: Read the data from path specified and store in pandas dataframe
 Output : dataframe 
'''
#########################################################################################
def load_dataset():
    df = pd.read_csv(RAW_DATA_PATH)
    return df


#########################################################################################
'''
 Input : DataFrame.
 Descripton: The three methods below impute values of columns height Restriction, 
 p4w new drivers and average ratings. All necessary details are given in
 EDA & Data Prepation ipynb notebook
 Output : dataframe 
'''
#########################################################################################

def impute_values_in_height_restriction_column(df):
    df['height_restriction']= df.height_restriction.fillna(180)
    return df

def impute_p4w_new_drivers(df):
    df['p4w_new_drivers'] =df['p4w_new_drivers'].fillna(0)
    return df

def impute_average_ratings(df):
    # Rounding off average ratings
    df['average_star_rating'] = df['average_star_rating'].round(decimals =0)

    # Numeric columns to be considered in the data
    num_col = [feature for feature in df.columns if df[feature].dtype in ['int64', 'float64']][3:]
    
    # Remaining features
    features_to_be_concatenated_later = list(set(df.columns).difference(set(num_col)))
    
    # Filtering on numeric columns
    df_filtered = df.filter(num_col, axis =1).copy()
    
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    transformed_data = scaler.fit_transform(df_filtered)
    df_knn = pd.DataFrame(transformed_data, columns = df_filtered.columns)
    
    # Define KNN imputer and fill missing values
    knn_imputer = KNNImputer(n_neighbors=3, weights='uniform', metric='nan_euclidean')
    df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df_knn), 
                                  columns=df_knn.columns)

    #Transforming the features back to scale
    df_imputed = pd.DataFrame(scaler.inverse_transform(df_knn_imputed), 
                              columns = df_filtered.columns)
    
    # Concating the imputed dataframe with remaining dataframe
    df_new = pd.concat([df[features_to_be_concatenated_later],df_imputed], axis=1)
    
    # Rounding the ratings
    df_new['average_star_rating'] = df_new['average_star_rating'].round(decimals =0)
    
    # Rearranging the columns
    df_new = df_new[df.columns]
    
    return df_new


#########################################################################################
'''
 Input : DataFrame.
 Descripton: This method needs is main method to do overall imputations
 Output : dataframe 
'''
#########################################################################################
def impute_missing_values(df):
    df = impute_values_in_height_restriction_column(df)
    df = impute_p4w_new_drivers(df)
    df = impute_average_ratings(df)
    return df


#########################################################################################
'''
 Input : DataFrame
 Description: Columns having data in string but describing about percentages are handled
 Output : DataFrame 
'''
#########################################################################################
def handling_columns_with_percent_sign(df):
    for col in COLUMNS_WITH_PERCENT_SIGN :
        df[col] = df[col].str.rstrip('%').astype('float') / 100.0
    return df


#########################################################################################
'''
 Input : List of Columns that need to be dropped
 Description: Drop all columns specified in the list
 Output : dataframe 
'''
#########################################################################################
def handle_rental_count(df):
    df['rental_count'] = df.count_first_rentals + df.count_repeat
    return df


#########################################################################################
'''
 Input : DataFrame
 Description: Assigns labels to Parking Types 
 Output : DataFrame with specified change
'''
#########################################################################################
def handle_parking_type(df):
    df.parking_type=df.parking_type.map({'self':0, 'valet':1, 'self-valet-assist':2})
    return df


#########################################################################################
'''
 Input : DataFrame
 Description: Handles columns that require special treatments
 Output : DataFrame with specified change
'''
#########################################################################################
def other_processings(df):
    df = handling_columns_with_percent_sign(df)
    df= handle_rental_count(df)
    df = handle_parking_type(df)
    return df


#########################################################################################
'''
 Input : List of Columns that need to be dropped
 Description: Drop all columns specified in the list
 Output : dataframe 
'''
#########################################################################################
def drop_unnecessary_columns(df):
    df.drop(columns = REDUNDANT_COLUMN_LIST, inplace = True)
    return df


#########################################################################################
'''
 Input : DataFrame
 Description: This method uses Label Encoding for transdorming  
              neighborhoods with labels
 Output : LabelEncoder object,DataFrame with specified change
'''
#########################################################################################
def handle_democratic_variable(df):
    # Label encoding reporting neighborhoods with sklearns preprocessing 
    le = preprocessing.LabelEncoder()

    # Fitting data to label encoder object
    le.fit(df.reporting_neighborhood)

    # Storing transformed data
    df.reporting_neighborhood=le.transform(df.reporting_neighborhood)

    np.save('reporting_neighborhoods.npy', le.classes_)
    return  df


#########################################################################################
'''
 Input : DataFrame
 Description: change datatypes of categorical columns to type category
 Output : DataFrame with specified change
'''
#########################################################################################
def handle_categorical_variables(df):
    # Assigning codes to categories in parking_type
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category").cat.codes
    return df


#########################################################################################
'''
 Input : DataFrame
 Description: change datatypes of numeric columns to type float64
 Output : DataFrame with specified change
'''
#########################################################################################
def handling_numerical_variables(df):
    numeric_columns = [col for col in df.columns if (col not in DEMOCRATIC_COLUMNS) and (col not in CATEGORICAL_COLUMNS)]
    df[numeric_columns] = df[numeric_columns].astype(np.float64)
    return df
    

#########################################################################################
'''
 Input : DataFrame
 Description: Method to change datatypes
 Output : DataFrame with specified change
'''
#########################################################################################
def change_datatypes(df):
    df = handle_categorical_variables(df)
    df = handling_numerical_variables(df)
    df = handle_democratic_variable(df)
    return df


#########################################################################################
'''
 Input : DataFrame
 Description: Write file in specified Output Path
 
'''
#########################################################################################
def write_file_to_folder(df):
    df.to_csv(OUTPUT_PATH+'processed_data.csv', index = False)




if __name__=='__main__':
    df = load_dataset()
    print('......Dataset loaded')

    df = impute_missing_values(df)
    print('......Missing values imputed')

    df = other_processings(df)
    print('......Other processings done')

    df = drop_unnecessary_columns(df)
    print('.....Unnecessary columns dropped')

    df = change_datatypes(df)
    print('......Changed DataTypes')

    write_file_to_folder(df)
    df = print('.....File written data/processed folder')
 















