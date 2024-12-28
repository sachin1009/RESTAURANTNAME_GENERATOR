import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

groq_api_key = "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"

llm = ChatGroq(
    temperature=0.7,
    groq_api_key=groq_api_key,
    model_name="llama-3.1-70b-versatile"
)

def generate_restaurant_name_and_items(cuisine):
    name_prompt = PromptTemplate.from_template(
        "Suggest a single fancy name for a {cuisine} restaurant. Provide only the name, without any explanation."
    )
    name_chain = name_prompt | llm | StrOutputParser()

    items_prompt = PromptTemplate.from_template(
        "Suggest 5 menu items for a {cuisine} restaurant named {restaurant_name}. Provide only the names of the dishes, separated by commas, without any explanations."
    )
    items_chain = items_prompt | llm | StrOutputParser()

    chain = RunnablePassthrough.assign(
        restaurant_name=name_chain
    ).assign(
        menu_items=lambda x: items_chain.invoke({"cuisine": x["cuisine"], "restaurant_name": x["restaurant_name"]})
    )

    return chain.invoke({"cuisine": cuisine})

st.title("Restaurant Name and Menu Generator")

cuisine = st.sidebar.selectbox("Pick a Cuisine", ("Indian", "Italian", "Mexican", "Chinese", "Japanese", "French", "Thai", "Greek", "Spanish", "American"))

if cuisine:
    with st.spinner("Generating restaurant name and menu..."):
        response = generate_restaurant_name_and_items(cuisine)
    
    st.header("Restaurant Name")
    st.subheader(response['restaurant_name'])
    
    st.header("Menu Items")
    menu_items = response['menu_items'].split(",")
    for item in menu_items:
        st.write("•", item.strip())