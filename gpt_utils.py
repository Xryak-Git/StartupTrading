import pandas as pd
import numpy as np
from datetime import timedelta
from .entity_detection import get_entity
from .spacy_key_words import *
import json
import ast
# from prompts import question_names, summary
import ast
import openai
import time
import concurrent.futures
import threading
import random
import json
import re
import os

path = '/Users/alexanderdemachev/PycharmProjects/strategy/My ML practice/aq/news_trading/data/'
single_thread = True
if single_thread:
    max_workers = 1
else:
    max_workers = 5
semaphore = threading.Semaphore(max_workers)

def get_all_tickers(path):
    files = os.listdir(path)
    tickers = []
    for file in files:
        if file.split('.')[0][-4:] == 'USDT':
            file = file.split('.')[0]
            tickers.append(file)
    return tickers

def replace_contractions(text):
    contractions = {
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "I would",
        "i'll": "I will",
        "i'm": "I am",
        "i've": "I have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "mightn't": "might not",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "we'd": "we would",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
        # Add any specific contractions or possessive forms you want to replace
    }

    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    return text




def filter_tickers(df):
    # This will print the unique data types present in the column
    df['ticker_detection'] = df['ticker_detection'].apply(lambda x: np.nan if x == {} else x)
    df.dropna(subset='ticker_detection', inplace=True)
    return df


def get_unique_coins(df):
    coins = df['ticker_detection'].unique()
    coins = [c.split(',') for c in coins]
    coins = [item for sublist in coins for item in sublist]
    coins = [c.replace(' ', '') for c in coins]
    coins = list(set(coins))
    print(coins)
    return coins


def clean_categories(df):
    categories = ['etf', 'financial', 'legal']
    for c in categories:
        df['gpt'] = np.where(df['gpt'].str.contains(c), c, df['gpt'])
    if df['gpt'].str.contains(',').any():
        print(df[df['gpt'].str.contains(',')])
        raise Exception('Still duplicated categories')
    return df


def spacy_categories(df, headline_col_name, target_categories, with_explanation=True):
    import spacy
    from spacy.matcher import Matcher

    # Initialize Spacy and the matcher
    nlp = spacy.load("en_core_web_lg")
    matcher = Matcher(nlp.vocab)

    # Function to add patterns to the matcher based on keywords
    def add_patterns(matcher, label, keywords):
        patterns = []
        for token in keywords:
            # Add pattern for lemmatization
            patterns.append([{"LEMMA": token.lower()}])
            # Add pattern for case-insensitive exact match
            patterns.append([{"LOWER": token.lower()}])
        matcher.add(label, patterns)

    keyword_dict = {name.split('_')[0]: obj for name, obj in globals().items() if isinstance(obj, list)}
    keyword_dict = {k: v for k, v in keyword_dict.items() if k in target_categories}

    # Add patterns to the matcher
    for k_name, k_list in keyword_dict.items():
        add_patterns(matcher, k_name, k_list)
    if with_explanation:
        def categorize_text(text):
            doc = nlp(str(text).lower())
            matches = matcher(doc)
            categories = {}  # Dictionary to store categories and their matched keywords

            for match_id, start, end in matches:
                rule_id = nlp.vocab.strings[match_id]  # Category
                matched_keyword = doc[start:end].text  # Matched keyword

                # Adding the matched keyword to the appropriate category
                if rule_id in categories:
                    categories[rule_id].add(matched_keyword)
                else:
                    categories[rule_id] = {matched_keyword}

            # Convert sets to lists for easier reading and manipulation
            for category in categories:
                categories[category] = list(categories[category])

            return categories

    else:
        def categorize_text(text):
            doc = nlp(str(text).lower())
            matches = matcher(doc)
            categories = set()  # Use a set to avoid duplicates
            for match_id, start, end in matches:
                rule_id = nlp.vocab.strings[match_id]
                categories.add(rule_id)
            return list(categories)

    df['spacy_category'] = df[headline_col_name].apply(categorize_text)
    return df


def remove_elements(elem_list, words_list):
    for l in elem_list:
        words_list = [w.replace(l, '').lower() for w in words_list]
    return set(words_list)


def categorize_as_duplicate(base_sentence, target_sentence, threshold_percentage):
    elem_to_delete = ['.', ',', "'", ':', '!']

    # Tokenize the sentences into words and convert to lowercase for case-insensitive comparison
    words_base = set(base_sentence.lower().split())
    words_target = set(target_sentence.lower().split())

    words_base = remove_elements(elem_to_delete, words_base)
    words_target = remove_elements(elem_to_delete, words_target)

    # Calculate the intersection of words
    common_words = words_base.intersection(words_target)

    # Calculate the percentage of common words relative to the number of unique words in the longer sentence
    unique_words_in_shorter_sentence = min(len(words_base), len(words_target))
    common_percentage = (len(common_words) / unique_words_in_shorter_sentence) * 100

    # Check if the percentage meets the threshold
    return 'duplicate' if common_percentage >= threshold_percentage else 'not duplicate'


def is_duplicate_to_any(base_sentence, other_sentences, threshold_percentage):
    for target_sentence in other_sentences:
        if categorize_as_duplicate(base_sentence, target_sentence, threshold_percentage) == 'duplicate':
            return f'Duplicate: {target_sentence}'
    return 'Not duplicate'


def check_for_duplicates(row, df, headline_col_name, threshold_percentage):
    # Filter the DataFrame for the past 7 days
    try:
        start_date = max(row.name - timedelta(days=1), df.index[0])
        other_sentences = df.loc[start_date:max(row.name - pd.Timedelta(seconds=1), df.index[0]),
                          headline_col_name].tolist()

        # Check if the sentence is a duplicate of any in the past 7 days
        return is_duplicate_to_any(row[headline_col_name], other_sentences, threshold_percentage)
    except:
        print('Problem with checking for duplicates, row:', row)
        return np.nan


def remove_short_headlines(df, headline_col_name):
    def count_unique_words(text):
        if isinstance(text, str):  # Check if the value is a string
            return len(set(text.split()))
        else:
            return 0

    df = df[df[headline_col_name].apply(count_unique_words) >= 3]
    return df


def remove_duplicate_messages(df, headline_col_name, threshold):
    df.drop_duplicates(subset=headline_col_name, inplace=True)
    df['is_duplicate'] = df.apply(lambda row: check_for_duplicates(row, df, headline_col_name, threshold), axis=1)
    df = df[df['is_duplicate'].str.lower() == 'not duplicate']
    return df


def pre_gpt_pipeline(df, headline_col_name, target_categories, filter_short_headlines=True, filter_duplicates=True,
                     ticker_detection=True, filter_categories=True):
    print('Adding ticker detection')
    entity_detector = get_entity(df[headline_col_name])
    df['ticker_detection'] = entity_detector.list_detection.values

    if filter_short_headlines:
        print('Removing short headlines, file len', len(df))
        df = remove_short_headlines(df, headline_col_name)
    if ticker_detection:
        print('Filtering crypto entities, file len', len(df))
        df = filter_tickers(df)
    if filter_duplicates:
        print('Removing duplicated messages, file len', len(df))
        df = remove_duplicate_messages(df, headline_col_name, 70)
    if filter_categories:
        print('Adding spacy categories')
        df = spacy_categories(df, headline_col_name, target_categories)

        print('Removing rows without categories, file len', len(df))
        df = df[df['spacy_category'] != {}]
    return df


def create_assistant(client, prompt, prompt_type):
    if prompt_type == 'summary':
        assistant = client.beta.assistants.create(
            name="News Summary Assistant",
            instructions=prompt,
            model="gpt-4-1106-preview"
        )
        print(f'Creating summary assistant for {prompt_type}')

    else:
        assistant = client.beta.assistants.create(
            name="News Categorization Assistant",
            instructions=prompt,
            model="gpt-4-1106-preview"
        )
        print(f'Creating assistant for {prompt_type}')

    return assistant


def create_summary(client, headlines):
    prompt = summary()

    assistant = create_assistant(prompt, prompt_type='summary')
    result_list = []
    for headline in headlines:
        try:
            thread = client.beta.threads.create()
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=headline
            )
            client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
            )
            time.sleep(5)  # Adjust as necessary

            messages_page = client.beta.threads.messages.list(thread_id=thread.id)
            for message in messages_page.data:
                if message.role == 'assistant' and message.content:
                    content = message.content[0]
                    if hasattr(content, 'text'):
                        result_list.append(content.text.value)
        except:
            continue

    return result_list


### SINGLE THREAD FOR TICKER ###


def send_message_no_semaphore(client, thread_id, assistant_id, headline, max_retries=5):
    try:
        retries = 0
        while retries < max_retries:

            client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=headline
            )
            client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
            )
            time.sleep(max(5, len(headline) // 5))

            messages_page = client.beta.threads.messages.list(thread_id=thread_id)
            for message in messages_page.data:
                if message.role == 'assistant' and message.content:
                    content = message.content[0]
                    if hasattr(content, 'text') and content.text.value.strip():
                        return headline, content.text.value

            retries += 1
            print('Rerunning for', headline)
            time.sleep(10)
    except Exception as e:
        print(f"An error occurred, retrying: {e}")
        time.sleep(30)
        return send_message(client, thread_id, assistant_id, headline)


def group_headlines_by_tokens(dataframe, headline_col, ticker_col):
    grouped = {}
    for index, row in dataframe.iterrows():
        ticker_data = row[ticker_col]

        # Check if ticker_data is a string representation of a dictionary
        if isinstance(ticker_data, str) and ticker_data.startswith('{'):
            try:
                entities = ast.literal_eval(ticker_data)
                if entities and isinstance(entities, dict):
                    for entity, tokens in entities.items():
                        for token in tokens:
                            if token not in grouped:
                                grouped[token] = []
                            grouped[token].append(row[headline_col])
            except Exception as e:
                print('Problem with headline when grouping by tokens')
                print(row[headline_col])
                print(ticker_data)
        elif isinstance(ticker_data, str):
            # Handle the case where ticker_data is a simple string token
            token = ticker
            if token not in grouped:
                grouped[token] = []
            grouped[token].append(row[headline_col])
        else:
            # Handle other cases, such as if ticker_data is neither a string dictionary nor a string token
            print('Unexpected data format in ticker_col')
            print(row[headline_col])
            print(ticker_data)
    return grouped


def process_token_headlines(client, assistant_id, token, headlines):
    responses = {}
    thread = client.beta.threads.create()  # Create a thread for each token
    thread_id = thread.id

    for headline in headlines:
        headline, response = send_message(client, thread_id, assistant_id, headline)
        responses[headline] = response

    return token, responses


def send_bulk_messages_single_thread(client, assistant_id, dataframe, headline_col, ticker_col):
    headlines_by_token = group_headlines_by_tokens(dataframe, headline_col, ticker_col)
    all_responses = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_token_headlines, client, assistant_id, token, headlines) for token, headlines
                   in headlines_by_token.items()]
        for future in concurrent.futures.as_completed(futures):
            try:
                token, responses = future.result()
                all_responses.update(responses)
            except Exception as e:
                print('Problem with processing token:', e)

    return all_responses


### STANDARD VERSION ###

def send_message(client, assistant_id, headline, max_retries=5):
    semaphore.acquire()
    try:
        retries = 0
        while retries < max_retries:

            thread = client.beta.threads.create()
            thread_id = thread.id

            client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=headline
            )
            client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
            )
            time.sleep(max(5, len(headline) // 5))

            messages_page = client.beta.threads.messages.list(thread_id=thread_id)
            for message in messages_page.data:
                if message.role == 'assistant' and message.content:
                    content = message.content[0]
                    if hasattr(content, 'text') and content.text.value.strip():
                        return headline, content.text.value

            retries += 1
            print('Rerunning for', headline)
            time.sleep(10)
    except Exception as e:
        print(f"An error occurred, retrying: {e}")
        time.sleep(30)
        return send_message(client, thread_id, assistant_id, headline)
    finally:
        semaphore.release()


def send_bulk_messages(client, assistant_id, headlines):
    responses = {}
    print('Creating thread for bulk messages')

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_headline = {executor.submit(send_message, client, assistant_id, headline): headline for headline in
                              headlines}
        for future in concurrent.futures.as_completed(future_to_headline):
            try:
                headline, response = future.result()
                responses[headline] = response
            except Exception as e:
                print('problem with headline')
                print(e)

    return responses


def merge_headline_with_tickers(row, message_col, ticker_col):
    return f'Headline: {row[message_col]}, \nCoins dictionary: {row[ticker_col]}'



### PROCESSING GPT RESULTS ###

def custom_json_parser(json_string):
    # Define the keys we expect in the JSON
    keys = ["Target Company Token", "Other Party", "Other Party Industry",
            "Other Party Size", "Alliance Type",
            "Entities Identified", "Primary Targets", "Impact Classifications"
            ]

    # Initialize an empty dictionary to hold the parsed values
    parsed_data = {}

    # Iterate over the keys and extract their values from the string
    for key in keys:
        # Create a regex pattern to find the key and its value
        pattern = rf'"{key}":\s*"(.*?)(?<!\\)",?'
        match = re.search(pattern, json_string)

        if match:
            # Extract the value, handling escaped quotes
            value = match.group(1).replace('\\"', '"')
            # Remove trailing comma if it's there
            value = re.sub(r',$', '', value.strip())
            parsed_data[key] = value
        else:
            # If the key is not found or value is missing, use null or an empty string
            parsed_data[key] = None

    # Special handling for the "Impact Classification" as it is a nested structure
    # Updated pattern to account for an optional trailing comma
    impact_pattern = r'"Impact Classification":\s*{\s*"Impact":\s*(\d+)\s*,?\s*},?'
    impact_match = re.search(impact_pattern, json_string)
    if impact_match:
        impact_value = int(impact_match.group(1))
        parsed_data["Impact Classification"] = {"Impact": impact_value}
    else:
        parsed_data["Impact Classification"] = {"Impact": None}

    return parsed_data

def split_json_objects(json_string):
    try:
        # Attempt to parse the entire string as a single JSON object
        obj = custom_json_parser(json_string)
        return [obj]
    except json.JSONDecodeError:
        # If it fails, proceed with the original logic for multiple objects
        objects = []
        brace_count = 0
        start_index = 0

        for i, char in enumerate(json_string):
            if char == '{':
                brace_count += 1
                if brace_count == 1:
                    start_index = i
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_index = i + 1
                    try:
                        obj = custom_json_parser(json_string[start_index:end_index])
                        objects.append(obj)
                    except json.JSONDecodeError:
                        # Handle the exception or print a warning
                        print("Warning: Invalid JSON detected")
                        print(json_string[start_index:end_index])
                        print('---')
        return objects

def fix_quotes_and_convert(json_string):
    # Pattern to match unescaped double quotes within the JSON string values
    pattern = r'(?<=:)\s*"(.*?)(?<!\\)"(?=\s*,|\s*})'

    # Function to replace unescaped double quotes with escaped ones inside the value
    def replace_func(match):
        # Escape all double quotes inside the value
        value = match.group(1).replace('"', '\\"')
        return f'"{value}"'

    # Fixing the inner double quotes
    fixed_json = re.sub(pattern, replace_func, json_string)

    # Convert the fixed string to dictionary
    return json.loads(fixed_json)


def process_classification_result(df, prompt_type, task_type, responses, headline_col):



    for headline, response in responses.items():
        response_cleaned = response.replace('```json', '').replace('```', '').replace('\n', '').strip()
        if response_cleaned:
            class_value = None
            if isinstance(response_cleaned, str):
                try:
                    response_dict = fix_quotes_and_convert(response_cleaned)
                    # response_dict = ast.literal_eval(response_cleaned)
                    if not isinstance(response_dict, dict):
                        print('Wrong response type')
                        print(headline)
                        print(response)
                        print('---')
                        continue
                except ValueError as e:
                    print(f"Error parsing string: {e}")
                    print(response_cleaned)
                    print('---')
                    class_value = response_cleaned.split(':')[1]
                    class_value = int(re.findall(r'\d+', class_value)[0])
            elif isinstance(response_cleaned, dict):
                pass
            else:
                raise Exception('Wrong result format returned for', response_cleaned)
            if prompt_type == 'simple_classification':
                for q_num, q_name in question_names.items():
                    df.at[df[df['Messages'] == headline].index[0], q_name] = response_dict.get(str(q_num), 0)
            elif task_type == 'classification':
                if class_value is not None:
                    df.loc[df[headline_col] == headline, f'{prompt_type}_{task_type}'] = class_value
                else:
                    try:
                        df.loc[df[headline_col] == headline, f'{prompt_type}_{task_type}'] = list(response_dict.values())[0]
                    except:
                        print('Problem with locing df for a headline')
                        print(headline)
                        print('---')
                    if len(response_dict) > 1:
                        df.loc[df[headline_col] == headline, f'{prompt_type}_{task_type}_reason'] = \
                            list(response_dict.values())[1]
            else:
                raise Exception('Wrong prompt or task type')
        else:
            print(f"Empty response received for '{headline}'.")


def process_sentiment_result(df, headline_col_name, time_col_name, prompt_type, responses, json_file_name):
    if responses == None:
        with open(json_file_name, 'r') as file:
            responses = json.load(file)
    else:
        with open(json_file_name, 'w') as file:
            json.dump(responses, file)

    results = []
    for init_headline, response in responses.items():
        response_cleaned = response.replace('```json', '').replace('```', '').replace('\n', '').strip()
        # response_cleaned_list = split_json_objects(response_cleaned)
        # for response_dict in response_cleaned_list:
        response_dict = ast.literal_eval(response_cleaned)
        if isinstance(response_dict, dict):
            try:
                # response_dict = ast.literal_eval(response_cleaned)
                if 'hack' in prompt_type:

                    classifications_dict = response_dict['Impact Classifications']
                    target_entities = list(classifications_dict.keys())
                    coins_dict = init_headline.split('Coins dictionary: ')[1]
                    entity_token_mapping = ast.literal_eval(coins_dict)

                    class_result = {}

                    for entity in target_entities:
                        for key, token_list in entity_token_mapping.items():
                            for token in token_list:
                                lower_entity = entity.lower()
                                lower_key = key.lower()
                                lower_token = token.lower()
                                if lower_key in lower_entity or lower_token in lower_entity:
                                    class_result[token] = classifications_dict[entity]

                elif 'partnership' in prompt_type:

                        target_token = response_dict['Target Company Token']
                        class_result = {}
                        class_result[target_token] = response_dict['Impact Classification']

            except Exception as e:
                print('Problem with', response_cleaned)
                print(e)
                print('---')
                continue
        else:
            raise Exception('Wrong result format returned for', response_cleaned)

        try:
            date = df.loc[df[headline_col_name] == init_headline, time_col_name].iloc[0]
        except:
            print('Headline not found in dataframe', headline)

        headline = init_headline.replace('Headline:', '')
        coins_dict_index = headline.find("Coins dictionary")
        headline = headline[:coins_dict_index].strip()

        for token, result in class_result.items():
            impact = result['Impact']
            if 'Reason' in result:
                reason = result['Reason']
                results.append([date, headline, prompt_type, token, impact, reason])
            else:
                results.append([date, headline, prompt_type, token, impact])
    if np.shape(results)[1] == 5:
        columns = ['date', 'headline', 'class', 'ticker', 'impact',]
    else:
        columns = ['date', 'headline', 'class', 'ticker', 'impact', 'reason']
    result = pd.DataFrame(results, columns=columns)
    return result


def process_entity_detection_result(df, headline_col_name, responses, json_file_name):
    if responses == None:
        with open(json_file_name, 'r') as file:
            responses = json.load(file)
    with open(json_file_name, 'w') as file:
        json.dump(responses, file)

    for headline, response in responses.items():
        response_cleaned = response.replace('```json', '').replace('```', '').replace('\n', '').strip()
        if isinstance(response_cleaned, str):
            # if 'Output:' in response_cleaned:
            #     response_cleaned = response_cleaned.replace('Output: ','')
            #     response_cleaned = replace_contractions(response_cleaned)
            try:
                response_dict = re.findall(r'\{.*?\}', response_cleaned)[0]
                response_dict = ast.literal_eval(response_dict)
            except Exception as e:
                print('Problem:', e)
                print(response_cleaned)
                continue
        elif isinstance(response_cleaned, dict):
            pass
        else:
            raise Exception('Wrong result format returned for', response_cleaned)
        df.loc[df[headline_col_name] == headline, 'entity_detection'] = str(response_dict)


def process_duplicate_result(df, responses, headline_col, time_col_name):
    results = []
    for ticker, headline_dict in responses.items():
        for headline, response in headline_dict.items():
            response_cleaned = response.replace('```json', '').replace('```', '').replace('\n', '').strip()
            if response_cleaned:
                try:
                    if isinstance(response_cleaned, str):
                        with_explanation = True
                        if len(response_cleaned) == 1:
                            duplicate_flag= int(response_cleaned)
                            with_explanation = False
                        else:
                            response_dict = ast.literal_eval(response_cleaned)
                    elif isinstance(response_cleaned, dict):
                        pass
                    else:
                        print('Wrong result format returned for', response_cleaned)
                        continue

                    date = df.loc[df[headline_col] == headline, time_col_name].iloc[0]

                    if with_explanation:
                        duplicate_flag = response_dict['Duplicate']
                        duplicate_reason = response_dict['Explanation']
                        results.append([date, ticker, headline, duplicate_flag, duplicate_reason])
                    else:
                        results.append([date, ticker, headline, duplicate_flag])
                except:
                    print('Problem with response')
                    print(response_cleaned)
                    print('---')

            else:
                print(f"Empty response received for '{headline}'.")
    result_shape = np.shape(results)
    if result_shape[1] == 4:
        columns = ['date', 'ticker', 'headline', 'spot_duplicates']
    else:
        columns = ['date', 'ticker', 'headline', 'spot_duplicates','spot_duplicates_reason']
    result = pd.DataFrame(results, columns=columns)
    return result
