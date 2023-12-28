## try to combine langchain and qdrant

in fact, both openai and google gemini works. in chinese, openai gpt-4 is far more better than gpt-3.5.

qdrant works, but when recreate collection, it shows time out all the time, and need to re run the function to make it work.

langchain have a good start and good ideas, but in the day of exploding ai technics, it seems a little bit lost.

## Try Gemini

test gemini chat...failed.  seems like langchain chat api not function in a  right way, like gemini never receive chat history and referenced documents.

## Learning

learn langchain from this project: https://github.com/alejandro-ao/ask-multiple-pdfs

and then try to port it to gemini api, but failed...

learnt a lot, streamlit is good and qdrant is finally work.

## todo

i'd still try to make gemini work, for it's free so far..., some days later maybe, give langchain some time to fix their codes :)

## UPDATE
gemini works...just like they don't.  in one word: after change model loader, it work a 'little', and almost none for chinese :(