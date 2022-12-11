# 
# Streamlit demonstration app.py
# Created at 1:26 AM 12/10/22 by Saket Joshi
#

import streamlit as st
import random, string
import importlib
import os
import sys
from io import StringIO
from contextlib import redirect_stdout

from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES
# from streamlit import caching
# from streamlit.legacy_caching import clear_cache
# from streamlit.script_runner import RerunException
# from streamlit_gallery.utils.readme import readme

# st.set_option("server.runOnSave", True)


examples = {
    "demo": """
print("Demo!")
""",

        "forward mode": """import numpy as np
from fab_ad.fab_ad_tensor import FabTensor, fab_session
from fab_ad.fab_admode import auto_diff
from fab_ad.constants import *


# scalar input; scalar output; forward ad
fab_session.clear()
x = FabTensor(value=3, identifier="x")
z = x ** 2 + 2 * x + 1
print(auto_diff(z, mode=AdMode.FORWARD))

""",

        "backward mode": """import numpy as np
from fab_ad.fab_ad_tensor import FabTensor, fab_session
from fab_ad.fab_admode import auto_diff
from fab_ad.constants import *

# multiple scalar input; scalar output; forward ad
fab_session.clear()
m = FabTensor(value=-4, identifier="m")
x = FabTensor(value=3, identifier="x")
y = FabTensor(value=-4, identifier="y")
z = x ** 2 + 2 * y ** 2
print(auto_diff(z, mode=AdMode.FORWARD))

""",

    }

def make_example(example_name, example_code):
    st.subheader(example_name)

    c1, c2 = st.columns([3, 1])

    with c1:
        content = st_ace(
            value=example_code,  # examples[example],
            placeholder="Add your code here",  # c2.text_input("Editor placeholder", value="Write your code here"),
            height=200,
            language="python",  # c2.selectbox("Language mode", options=LANGUAGES, index=121),
            theme="clouds",  # c2.selectbox("Theme", options=THEMES, index=35),
            # keybinding=c2.selectbox("Keybinding mode", options=KEYBINDINGS, index=3),
            # font_size=c2.slider("Font size", 5, 24, 14),
            # tab_size=c2.slider("Tab size", 1, 8, 4),
            show_gutter=True,  # c2.checkbox("Show gutter", value=True),
            show_print_margin=False,  # c2.checkbox("Show print margin", value=False),
            wrap=True,  # c2.checkbox("Wrap enabled", value=False),
            auto_update=True,  # c2.checkbox("Auto update", value=False),
            readonly=False,  # c2.checkbox("Read-only", value=False),
            min_lines=45,
            key=example_name,
        )
        # content.setValue(examples[example_selection])
        # content = examples[example]
    with c2:
        # Create a run button
        run = st.button("Run", key=example_name+"_button")
        if run and content:
            print(f"Running code... \n {content}")
            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()
            exec(content)
            sys.stdout = old_stdout
            output = redirected_output.getvalue()
            st.subheader("Output")
            st.text(output)

    st.markdown("---")



def main():
    # Set the page title
    st.set_page_config(page_title="Fab-AD Usage", page_icon=":snake:", layout="wide")

    # Set the page header
    st.title("Fab-AD - Automated differentiation library")
    st.markdown("## A library for automatic differentiation")

    # Set the page subheader
    st.markdown("### Usage examples")

    # make_example("Forward mode", examples["forward mode"])
    for example in examples:
        make_example(example, examples[example])


    # c1, c2 = st.columns([3, 1])
    #
    # c2.subheader("Select an example")
    #
    # example = "forward mode"

    # with c2:
    #     example = st.selectbox("", list(examples.keys()))

    # with c1:
    #     # example_selection = c2.selectbox("", list(examples.keys()))
    #     # caching.clear_cache()
    #     # clear_cache()
    #
    #     content = st_ace(
    #         value=c2.text_input("examples", value="Write your code here"), # examples[example],
    #         placeholder="Add your code here", # c2.text_input("Editor placeholder", value="Write your code here"),
    #         height=200,
    #         language="python", # c2.selectbox("Language mode", options=LANGUAGES, index=121),
    #         theme="clouds", # c2.selectbox("Theme", options=THEMES, index=35),
    #         # keybinding=c2.selectbox("Keybinding mode", options=KEYBINDINGS, index=3),
    #         # font_size=c2.slider("Font size", 5, 24, 14),
    #         # tab_size=c2.slider("Tab size", 1, 8, 4),
    #         show_gutter=True, # c2.checkbox("Show gutter", value=True),
    #         show_print_margin=False, # c2.checkbox("Show print margin", value=False),
    #         wrap=True, #c2.checkbox("Wrap enabled", value=False),
    #         auto_update=True, #c2.checkbox("Auto update", value=False),
    #         readonly=False, # c2.checkbox("Read-only", value=False),
    #         min_lines=45,
    #         key="ace",
    #     )
    #     # content.setValue(examples[example_selection])
    #     # content = examples[example]
    #     if content:
    #         st.subheader("Content")
    #         st.text(content)



        # if example:
        #     raise RerunException
        # st.experimental_rerun()



if __name__ == "__main__":
    main()

    #
    # # Create ace editor
    # content = st_ace(
    #     value = example_1,
    #     placeholder="Enter your code here",
    #     language="python",
    #     theme="monokai",
    #     # key="ace",
    #     height=200,
    #     font_size=14,
    #     tab_size=4,
    #     # show_gutter=False,
    #     # show_print_margin=True,
    #     wrap=True
    # )
    #
    # # Create 2 buttons in a single row
    # # col1, col2 = st.columns(2)
    #
    # # Show code output when button is clicked
    # print("Code output:", content)
    # print("Output:", eval(content))
    # output = exec(content)
    # output
    # if run:
    #     # with st.echo():
    #     print(content)
            # exec(content)


    #
    #
    # strategy_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    # with open(strategy_name + '.py', 'w') as the_file:
    #     the_file.write(content)
    # TestStrategy = getattr(importlib.import_module(strategy_name), 'TestStrategy')
    #
    # # do stuff
    # if os.path.exists(strategy_name + '.py'):
    #     os.remove(strategy_name + '.py')
    # else:
    #     print("The file does not exist")
