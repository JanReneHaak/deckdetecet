import pandas as pd
import ast
import io
import base64
from fastapi.responses import JSONResponse
import matplotlib.pyplot as plt

# Function to get the price history of a card
def price_history(card_name: str, set: str, df_price: pd.DataFrame):
    plt.switch_backend('Agg')
    try:
        # Load the price dataframe
        df_price.iloc[:, 3:] = df_price.iloc[:, 3:].applymap(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        # Check if the card_name is in the df prises
        target = df_price [(df_price ["name"] == card_name.lower()) & (df_price ["set"] == set)].iloc[0,:]

        # Generate x and y variables for plotting (generating a dataframe to use pandas plotting)
        x = target.index[3:]
        y = target[3:].apply(lambda x: x.get("usd"))
        y = y.fillna(0)
        y = y.astype("float32")
        price_data = pd.DataFrame({"date": x, "price": y})

        # Plot the price history using Pandas Plotting and get the figure into am actual variable
        t = price_data.plot.line(color = "blue", legend = False, marker = "o", figsize = (16,8), fontsize = "large")
        t.set_ylabel('Price in USD', fontsize = "x-large")
        t.set_xlabel('Date', fontsize = "x-large")
        t.set_title(f"Recent sales for {card_name} on tcgplayer.com", fontsize = "xx-large")
        plot = t.get_figure()

        # Temporary store the image in memory (not on the disk!)
        buffer = io.BytesIO()
        plot.savefig(buffer, format='png', bbox_inches='tight')  # Save as PNG format -> Transparent background
        buffer.seek(0)
        plt.close('all')

        # Encode the binary data to a Base64 string
        price_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()  # Clean up the buffer/temporary memory storage

        # Return the image in a JSON response
        return JSONResponse(content={"price": price_plot})

    except Exception as e:
        return JSONResponse(content={"error": f"❌ Error generating price history: {str(e)}"})
