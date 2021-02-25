# Libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Import data σε pandas dataframes
# Τα csv αρχεία χωρίζουν τις στήλες με tab
# Αφαιρούμε την 1η στήλη επειδή είναι ίδια με το index που δημιουργεί το pandas
movies_df = pd.read_csv('data/movies.csv', sep='\t')
movies_df = movies_df.drop(movies_df.columns[0], axis=1)

users_df = pd.read_csv('data/users.csv', sep='\t')
users_df = users_df.drop(users_df.columns[0], axis=1)

ratings_df = pd.read_csv('data/ratings.csv', sep=';')
ratings_df = ratings_df.drop(ratings_df.columns[0], axis=1)

# Μετατρέπουμε τις στήλες gender και age_desc του users_df σε categorical data για να μπορούμε χρησιμοποιήσουμε την αριθμητική τους αναπαράσταση (cat.codes)
# (Ο KMeans δεν μπορεί να πάρει categorical data οπότε πρέπει να τα μετατρέψουμε σε αριθμούς)
users_df.gender = users_df.gender.astype('category')
users_df.age_desc = users_df.age_desc.astype('category')

# Unique genres χρησιμοποιώντας την explode() αφού τα χωρίσουμε σε κάθε |
unique_genres = movies_df["genres"].str.split('|').explode().unique()

# Συνάρτηση που ζητάει το id του χρήστη για τον οποίο θέλουμε να δώσουμε προτάσεις
def get_input_id():
    user_id = int(input("Enter user ID (-1 if you want to quit): \n"))
    return user_id

# Συνάρτηση που φτιάχνει και επιστρέφει το array που θα δώσουμε στον KMeans
def create_kmeans_dataset():
    user_ids = np.array(users_df.user_id) # Πάιρνουμε array με όλα τα ids από το users dataframe
    kmeans_complete_dataset = [] # Initialize το array

    # Loop για κάθε user και δημιουργία array με τα ratings του χρήστη σε κάθε είδος ΚΑΙ τα βασικά χαρακτηριστικά του (age, occupation etc...)
    for user_id in user_ids:
        # print για να εμφανίσουμε το progress (χρειάζεται λίγη ώρα για να κάνει το analyze)
        if (user_id % 1000 == 0):
            print('...+1000 users have been analyzed... Please wait...')

        kmeans_avg_ratings_dataset = get_average_user_ratings_by_genre(user_id) # παίρνουμε το array με τα average ratings του user ανά είδος ταινίας
        kmeans_user_info_dataset = get_user_info(user_id) # παίρνουμε το array με τα χαρακτηριστικά του user (gender, occupation, age_desc)

        # Συνδυάζουμε σε ένα array τα average ratings per genre και τα χαρακτηριστικά
        # Βάζουμε το array αυτό στο kmeans_complete_dataset το οποίο είναι ένα array of arrays με τα στοιχεία κάθε user
        kmeans_curr_user_dataset = np.append(kmeans_avg_ratings_dataset, kmeans_user_info_dataset)
        kmeans_complete_dataset.append(kmeans_curr_user_dataset)

    return kmeans_complete_dataset

# Συνάρτηση που επιστρέφει array με τις average βαθμολογίες κάθε είδους αναλόγως το χρήστη (χρειαζόμαστε το id του)
def get_average_user_ratings_by_genre(id):
    ratings_curr_user_df = ratings_df[ratings_df.user_id == id] # παίρνουμε τα ratings του συγκεκριμένου χρήστη από το ratings dataframe
    curr_user_merged_df = movies_df.merge(ratings_curr_user_df, on='movie_id', how='right') # κάνουμε merge με το movies dataframe (right merge για να πάρουμε μόνο τις ταινίες που έχει βαθμολογήσει ο χρήστης)
    ratings_per_genre_curr_user_df = curr_user_merged_df.assign(Genre=curr_user_merged_df.genres.str.split(r'|')).explode('Genre') # split τα genres σε διαφορετικά rows
    avg_ratings_per_genre_curr_user = ratings_per_genre_curr_user_df.groupby('Genre').rating.mean() # παίρνουμε τα average ratings με τη συνάρτηση mean() αφού πρώτα ομαδοποιήσουμε τις ταινίες ανά είδος
    avg_ratings_per_genre_curr_user = avg_ratings_per_genre_curr_user.reindex(unique_genres).fillna(0) # βάζουμε όλα τα genres και ορίζουμε το average rating ενός είδους ίσο με το 0 εάν ο χρήστης δεν έχει βαθμολογήσει καμία ταινία του συγκεκριμένου είδους
    avg_ratings_per_genre_curr_user.sort_index(inplace=True) # sort για να είναι όλες οι βαθμολογίες των χρηστών με την ίδια σειρά
    return np.array(avg_ratings_per_genre_curr_user.values) # επιστρέφουμε τα average ratings για κάθε είδος με τη μορφή ενός np.array

# Συνάρτηση που επιστρέφει array με τα χαρακτηριστικά ενός χρήστη (θα τα εισάγουμε στο kmeans dataset)
def get_user_info(id):
    curr_user_data = users_df[users_df.user_id == id] # παίρνουμε το row από το users dataframe με βάση το user_id
    user_info = np.append(curr_user_data.gender.cat.codes, curr_user_data.occupation) # εφόσον έχουμε ορίσει το gender ως categorical data παίρνουμε τον κωδικό του. Επίσης παίρνουμε το occupation που είναι ήδη κωδικοποιημένο
    user_info = np.append(user_info, curr_user_data.age_desc.cat.codes) # εφόσον έχουμε ορίσει το age_desc ως categorical data παίρνουμε τον κωδικό του
    return user_info

# Συνάρτηση που δημιουργεί και επιστρέφει dataframe με το id του χρήστη και το cluster στο οποίο ανήκει
def cluster_users(cluster_array):
    user_ids = np.array(users_df.user_id) # Πάιρνουμε array με όλα τα ids από το users dataframe
    users_clustered = pd.DataFrame(columns=['user_id', 'cluster']) # initialize dataframe με 2 columns -> user_id και cluster στο οποίο ανήκει ο συγκεκριμένος χρήστης

    i = 0 # το χρησιμοποιούμε για να κάνουμε iterate τις τιμές του cluster_array (δηλαδή τα predictions από τον kmeans)
    # Loop τα ids των χρηστών
    for id in user_ids:
        users_clustered = users_clustered.append({'user_id': id, 'cluster': cluster_array[i]}, ignore_index=True) # για κάθε χρ΄΄ηστη συμπληρώνουμε το id του και την ομάδα στην οποία ανήκει σύμφωνα με τις προβλέψεις του kmeans
        i += 1

    return users_clustered

# Συνάρτηση που επιστρέφει dataframe με τις ταινίες τις οποίες δεν έχει βαθμολογήσει ένας χρήστης
def get_user_incomplete_ratings(id):
    user_complete_ratings = ratings_df[ratings_df['user_id'] == id] # dataframe με όλα τα ratings του χρήστη με βάση το id του
    user_ratings_merged = movies_df.merge(user_complete_ratings, on='movie_id', how='left').drop(['user_id', 'timestamp', 'genres'], axis=1) # merge *left* με το movies dataframe για να πάρουμε το σύνολο των ταινιών (ακόμη και αυτές που δεν έχει βαθμολογήσει)
    user_incomplete_ratings = user_ratings_merged[user_ratings_merged.rating.isna()] # παίρνουμε μόνο τις ταινίες στις οποίες το rating είναι NaN

    return user_incomplete_ratings

# Συνάρτηση που επιστρέφει τον αριθμό του cluster στον οποίο έχει προβλέψει ο αλγόριθμος ότι ανήκει ενας χρήστης (χρειαζόμαστε το id του)
def get_user_cluster_num(id):
    return clusters_df[clusters_df['user_id']==id].cluster.values[0]

# Συνάρτηση που επιστρέφει τις βαθμολογίες όλων των χρηστών που ανήκουν σε ένα συγκεκριμένο cluster
def get_related_users_ratings(cluster_num):
    user_ids_same_cluster = clusters_df[clusters_df.cluster == cluster_num].user_id.values # user_ids στο ίδιο cluster με τον χρήστη
    related_users_ratings = pd.DataFrame()  # αρχικοποιούμε το dataframe που θα επιστρέψουμε

    # Loop τα user ids των χρηστών που ανήκουν στο ίδιο cluster
    for user_id in user_ids_same_cluster:
        current_ratings = ratings_df[ratings_df['user_id']==user_id].drop(['timestamp'], axis=1) # παίρνουμε τα ratings του συγκεκριμένου χρήστη
        related_users_ratings = related_users_ratings.append(current_ratings) # βάζουμε τα ratings στο dataframe που θα επιστρέψουμε

    # Αριθμός βαθμολογιών για κάθε ταινία
    movie_id_counts = related_users_ratings.movie_id.value_counts()
    # Ταινίες με τουλάχιστον 10 βαθμολογίες από διαφορετικούς χρήστες
    frequently_rated_movies = movie_id_counts[movie_id_counts >= 10]
    # array με ids ταινιών με τουλάχιστων 10 βαθμολογίες από διαφορετικούς χρήστες (frequently_rated_movies είναι τ΄ύπου pandas.Series οπότε για να πάρουμε τα ids παίρνουμε όλα τα indexes του series)
    frequently_rated_movies_ids = frequently_rated_movies.index

    # Επιστρέφουμε μόνο τα rating των ταινιών τις οποίες έχουν βαθμολογήσει τουλάχιστον 10 διαφορετικοί χρήστες
    related_users_ratings = related_users_ratings[related_users_ratings.movie_id.isin(frequently_rated_movies_ids)]
    return related_users_ratings

# Συνάρτηση που επιστρέφει dataframe με βαθμολογίες για τις ταινίες που ΔΕΝ έχει βαθμολογήσει ένας χρήστης
# Για κάθε ταινία χρησιμοποιούμε την average βαθμολογία που έχουν βάλει όλοι οι χρήστες οι οπο΄ίοι βρίσκονται στην ίδια ομάδα με το χρήστη για τον οποίο θέλουμε να κάνουμε προτάσεις ταινιών
def update_user_ratings(not_rated_df, related_users_ratings_df):
    related_users_avg_ratings_per_movie = related_users_ratings_df.groupby('movie_id').rating.mean() # Βρίσκουμε το average κάθε ταινίας
    user_updated_ratings = not_rated_df.set_index('movie_id').rating.fillna(related_users_avg_ratings_per_movie).reset_index() # Γεμίζουμε το column rating καθε ταινίας με το average rating από το παραπάνω dataframe
    return user_updated_ratings

# Συνάρτηση που τυπώνει το τελικό αποτέλεσμα
def print_results(results_df):
    results_df = results_df.merge(movies_df, on='movie_id', how='left') # Merge με το movies dataframe για να έχουμε και τους τίτλους των ταινιών (merge left για να πάρουμε μόνο τις ταινίες με rating)
    results_df = results_df.sort_values(by='rating', ascending=False).reset_index() # Sort τις ταινίες με βάση το rating
    
    print('Recommending 10 movies:\n')
    # Εμφανίζουμε κάθε φορά των τίτλο τις ταινίας και το σκορ της
    for i in range(10):
        current_title = results_df['title'].iloc[i]
        current_score = round(results_df['rating'].iloc[i], 1)
        print(f"{i+1} -> {current_title} | rating: {current_score}")


### ----------- ΑΡΧΗ ΠΡΟΓΡΑΜΜΑΤΟΣ -----------  ###
# Implement του KMeans
print('Preparing Data...')
X = create_kmeans_dataset() # δημιουργία dataset
kmeans = KMeans(n_clusters=5) # ζητάμε από τον kmeans να χρησιμοποιήσει 5 ομάδες για να χωρίσει τους χρήστες (αυθαίρετα, αλλά δεν είναι απαραίτητο ότι όσο μεγαλώνει ο αριθμός μεγαλώνει και η ακρίβεια του αποτελέσματος)
kmeans.fit(X)

# Predictions της ομάδας κάθε user
y_kmeans = kmeans.predict(X)
clusters_df = cluster_users(y_kmeans) # δημιουργούμε το dataframe με τους users και το cluster στο οποίο ανήκουν

# Main Loop
user_id = get_input_id() # Ζητάμε το id του χρήστη για το οποίο θα δώσουμε προτάσεις
while (user_id != -1):
    print(f'\nPlease wait while we try to find the best matches for you! (user_id:  {user_id})\n')

    # Βρίσκουμε το cluster στο οποίο ανήκει ο συγκεκριμένος χρήστης
    cluster_num = get_user_cluster_num(user_id)

    # Παίρνουμε τις βαθμολογίες όλων των χρηστών που βρίσκονται στην ίδια ομάδα
    related_users_ratings_df = get_related_users_ratings(cluster_num)

    # Παίρνουμε τις ταινίες τις οποίες ΔΕΝ έχει βαθμολογησει ο χρήστης
    not_rated_df = get_user_incomplete_ratings(user_id)
    
    # Κάνουμε update τις ταινίες που δεν έχει βαθμολογήσει ο χρήστης δίνοντας ως τιμή το average rating της κάθε ταινίας (την οποία έχουν βαθμολογήσει παρόμοιοι χρήστες)
    user_updated_ratings = update_user_ratings(not_rated_df, related_users_ratings_df)

    # Τέλος, εμφανίζουμε τις προτάσεις που κάνει το σύστημα
    print_results(user_updated_ratings)

    # Ζητάμε καινούριο id για να δώσουμε προτάσεις σε άλλο χρήστη
    user_id = get_input_id()
