import re
import pandas as pd
import ast

"""
This python file reads business info and reviews csv files. It also provides a series of functions
that are used to query the business and review data.
"""

business_info_df = pd.read_csv("csv_files/business_info_all.csv")
#Business categories are stored as one string. Convert to a list of strings.
business_info_df.category = business_info_df.category.map(ast.literal_eval)

reviews_df = pd.read_csv("csv_files/reviews_all.csv")
#Convert date field to pandas datetime type.
reviews_df.date = pd.to_datetime(reviews_df.date)

def format_name(business_name):
    """
    This function removes dashes and capitalizes each word in a text string.
    It is used in this project to make the business_index look pretty for Tableau 
    and for simple search queries.
    """
    return business_name.replace('-',' ').title()

business_info_df.biz_id = list(map(format_name,business_info_df.biz_id))
reviews_df.business_index = list(map(format_name,reviews_df.business_index))

def get_businesses_by_category(chosen_category):
    """
    This function returns all businesses in business_info_df for a particular category.
    Substrings of categories will also find the selected category.
    INPUTS:
    chosen_category = text string to match against category list in business_info_df category feature.
    OUTPUT:
    business_info_df selection based on chosen category.
    """
    chosen_category = chosen_category.lower()
    #Check if it is a substring also
    return business_info_df[ 
        [True in [bool(re.search(chosen_category, category.lower())) for category in category_list] 
         for category_list in business_info_df.category] ]
    #return business_info_df[[chosen_category in category for category in business_info_df.category]]

def get_reviews_by_category(chosen_category):
    """
    This function returns all reviews for a particular category.
    INPUTS:
    chosen_category = text string to match against category list in business_info_df category feature.
    OUTPUT:
    reviews_df selection based on chosen category.
    """
    #Select businesses
    biz_ids_for_category = get_businesses_by_category(chosen_category)['biz_id']
    #Select Reviews for businesses
    return reviews_df[[biz_id in biz_ids_for_category.values for biz_id in reviews_df.business_index.values]]

def get_businesses_by_name(search_name):
    """
    This function returns all businesses in business_info_df for a business name.
    Substrings of business name will also find the selected businesses.
    INPUTS:
    search_name = text string to match against business name in business_info_df.
    OUTPUT:
    business_info_df selection based on chosen search name.    
    """
    search_name = search_name.lower()
    #Check if it is a substring also
    return business_info_df[ 
        [bool(re.search(search_name, biz_name.lower())) for biz_name in business_info_df.name] ] 

def get_reviews_for_businesses(biz_id_list):
    """
    This function returns all reviews for a list of business_index's.
    INPUTS:
    biz_id_list = list of text strings representing business indexes to select reviews on.
    OUTPUT:
    reviews_df selection based on chosen businesses.
    """
    return reviews_df[[biz_id in biz_id_list for biz_id in reviews_df.business_index.values]]


def get_subset_list(min_businesses_per_category, min_reviews_per_business, min_reviews_per_category, verbose=True):
    """
    This function will return a list of business indexes and a list of categories that meet the 
    minimums specified by the function parameters. Setting verbose will print stats.
    INPUTS:
    min_businesses_per_category = Only return categories that have the minimum number of businesses.
    min_reviews_per_business    = Only return businesses that have the minimum number of reviews.
    min_reviews_per_category    = Only return categories that have the minimum number of reviews.
    OUTPUT:
    biz_list, category_list = list of businesses and list of categories.
    """
    #Get a list of all categories across businesses in business_info_df
    all_categories = []
    for item in business_info_df.category:
        for cat in item:
            all_categories.append(cat)
    all_categories = list(set(all_categories))
    
    category_list = []
    biz_list = []
    
    #Loop through categories and test each condition specified. Populate biz_list and category_list.
    for category in all_categories:
        all_businesses = get_businesses_by_category(category)['biz_id'].values
        selected_businesses = []
        for biz in all_businesses:
            if len(get_reviews_for_businesses([biz])) >= min_reviews_per_business:
                selected_businesses.append(biz)
        num_businesses = len(selected_businesses)
        num_reviews = len(get_reviews_for_businesses(selected_businesses))
        if num_businesses > min_businesses_per_category and num_reviews > min_reviews_per_category:
            category_list.append(category)
            biz_list.extend(selected_businesses)
            if verbose:
                tabs = "\t\t\t"
                if (len(category) > 7):
                    tabs = "\t\t"
                print(f"{category: <{20}} {num_businesses: >{3}} places,  {num_reviews} reviews")
                
    biz_list = list(set(biz_list))
    
    if verbose:
        print("")
        print(len(category_list),"categories.")
        print(len(biz_list),"unique businesses represented in these categories.")
        print(len(get_reviews_for_businesses(biz_list)),"unique reviews represented in these categories.")
        
    return biz_list, category_list