mkdir -p ~/.streamlit/

echo "\
[server]\n\
enableCORS=false\n\
port = $PORT\n\
\n\
[theme]\n\
primaryColor=\"#fff\"\n\
backgroundColor=\"#0e1117\"\n\
secondaryBackgroundColor=\"#31333F\"\n\
textColor=\"#fafafa\"\n\
" > ~/.streamlit/config.toml