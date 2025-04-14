import streamlit as st

# Function to train the model with the provided data
def train_model(task_desc, github_url, options, train_percent, test_val_percent, model_choice):
    # Concatenate the data into one string
    training_data = (
        f"Task description: {task_desc}\n"
        f"GitHub Dataset URL: {github_url}\n"
        f"Selected data options: {', '.join(options)}\n"
        f"Training percentage: {train_percent}%\n"
        f"Testing + Validation percentage: {test_val_percentage}%\n"
        f"Model chosen: {model_choice}\n"
    )

    st.write("Training data string:")
    st.text(training_data)

    st.success("Model training has started!")
    

st.title("Machine Learning Model Trainer")

text_input = st.text_input("Enter the task description", "train model")

github_url = st.text_input("Enter GitHub dataset link", "")

options = st.multiselect(
    "Select the data options (You can choose multiple)",
    ["Training", "Testing", "Validation"]
)

train_test_split = st.slider(
    "Select the percentage split for training vs testing and validation",
    min_value=10,
    max_value=90,
    value=70,
    step=5
)

test_val_percentage = 100 - train_test_split

st.write(f"Training data: {train_test_split}%")
st.write(f"Testing + Validation data: {test_val_percentage}%")

model = st.selectbox(
    "Choose a model",
    options=["CNN", "RNN", "SVM", "Random Forest", "Auto Model"],
    index=4
)

run_button = st.button("Run")

if run_button:
    train_model(
        text_input,
        github_url,
        options,
        train_test_split,
        test_val_percentage,
        model
    )
