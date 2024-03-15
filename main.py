import pandas as pd
from embeddings_utils import distances_from_embeddings
from openai import OpenAI
import numpy as np
from ast import literal_eval


def create_context(question, df, client, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    cur_len = 0
    returns = []

    # Get the embeddings for the question
    q_embeddings = client.embeddings.create(input=question, model='text-embedding-ada-002').data[0].embedding

    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    client,
    model="gpt-3.5-turbo",
    question="Hola",
    max_len=3000,
    size="ada",
    debug=False,
    max_tokens=800
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        client,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"Eres un asistente que tiene que responder info basado en este contexto: {context}"},
                {"role": "user", "content": f"{question}"}
            ],
            max_tokens=max_tokens,
            temperature=0,
            top_p=1,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(e)
        return ""

def main():
    client = OpenAI()
    df = pd.read_csv('processed/embeddings.csv', index_col=0)

    # convert string array to array
    embeddings = []
    for value in df['embeddings'].values:
        embeddings.append(literal_eval(value))
    df['embeddings'] = embeddings

    while True:
        question = input()
        print("\U0001f600", answer_question(
            df,
            client=client,
            model="gpt-4-turbo-preview",
            max_len=5000,
            question=question,
            debug=False))

if __name__ == "__main__":
    main()



