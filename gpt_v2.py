
from utils.prompts import *
from utils.gpt_utils import *

# Initialize the OpenAI client
client = openai.OpenAI(api_key='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')


prompt_types = ['partnership']
task_type = 'classification'
save_files = True
with_explanation = False
run_gpt = False
date_col = 'date'
headline_col = 'headline'
ticker_col = 'ticker_detection'
sent_col_name = 'headlines_ticker_detection'


def main():
    df = pd.read_csv(path + "['partnership']_entity_detection_tie_twitter.csv", parse_dates=[date_col])
    df = df[df['entity_detection'] != '{}']

    sentiment_list = []

    for prompt_type in prompt_types:

        responses = None
        if task_type == 'entity_detection':
            json_file_name = path  + f'entity_detection_tie_noTwitter_2023.json'
        else:
            json_file_name = path +f'{prompt_type}_{task_type}_tie_twitter.json'

        if task_type == 'sentiment':
            print('Processing for', prompt_type)
            prompt_df = df[df[f'{prompt_type}_classification'] == 1].copy(deep=True)
            prompt_df[sent_col_name] = prompt_df.apply(
                lambda x: merge_headline_with_tickers(x, headline_col, 'entity_detection'), axis=1)
            print('Total number of headlines', len(prompt_df))
        elif task_type == 'entity_detection':
            df[sent_col_name] = df.apply(
                lambda x: merge_headline_with_tickers(x, headline_col, 'ticker_detection'), axis=1)
            mod_headlines = df[sent_col_name].tolist()
            print('Total number of headlines', len(mod_headlines))
        elif task_type == 'classification':
            print('Processing for', prompt_type)
            headlines = df[headline_col].tolist()
            print('Total number of headlines', len(headlines))
        elif task_type == 'spot_duplicates':
            print('Total number of headlines', len(df))
        else:
            raise ('Wrong task type')

        if run_gpt:
            # Create prompt
            if task_type == 'classification':
                prompt = classification(prompt_type, with_explanation=with_explanation)
            elif task_type == 'sentiment':
                prompt = sentiment(prompt_type, with_explanation=with_explanation)
            elif task_type == 'entity_detection':
                prompt = detect_tickers(with_explanation=with_explanation)
            elif task_type == 'spot_duplicates':
                prompt = spot_duplicates()

            # Create assistand and send messages

            assistant = create_assistant(client, prompt, prompt_type=task_type)

            if task_type == 'sentiment':
                responses = send_bulk_messages_single_thread(client, assistant.id, prompt_df, sent_col_name, ticker_col)
            elif task_type == 'spot_duplicates':
                responses = send_bulk_messages_single_thread(client, assistant.id, df, headline_col, ticker_col)
            else:
                responses = send_bulk_messages(client, assistant.id, headlines)

            if task_type == 'simple_classification':
                for q_name in question_names.values():
                    df[q_name] = np.nan
            elif task_type == 'entity_detection':
                df[f'{task_type}'] = np.nan
            else:
                df[f'{prompt_type}_{task_type}'] = np.nan

        if not save_files:
            if with_explanation:
                for headline, response in responses.items():
                    print(headline)
                    print(response)
                    print('---')

        # Store results
        if save_files:

            if responses == None:
                with open(json_file_name, 'r') as file:
                    responses = json.load(file)
            else:
                with open(json_file_name, 'w') as file:
                    json.dump(responses, file)

            if task_type == 'sentiment':
                result_df = process_sentiment_result(prompt_df, sent_col_name, date_col, prompt_type, responses,
                                                     json_file_name)
                sentiment_list.append(result_df)
            elif task_type == 'classification':
                process_classification_result(df, prompt_type, task_type, responses, headline_col)
            elif task_type == 'entity_detection':
                process_entity_detection_result(df, sent_col_name, responses, json_file_name)
            elif task_type == 'spot_duplicates':
                df = process_duplicate_result(df, responses, headline_col,date_col)
        if task_type == 'entity_detection' or task_type == 'spot_duplicates':
            break

        if task_type == 'sentiment' and save_files:
            df = pd.concat(sentiment_list)
    if save_files:
        df.to_csv(path + f'{prompt_types}_{task_type}_tie_Twitter.csv', index=False)


if __name__ == "__main__":
    main()

