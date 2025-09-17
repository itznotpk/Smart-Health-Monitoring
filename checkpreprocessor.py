import joblib

# Load preprocessor
preprocessor = joblib.load('preprocessor.joblib')

# Print the full preprocessor object
print(preprocessor)

# Inspect each transformer
for name, transformer, columns in preprocessor.transformers_:
    print(f"Transformer name: {name}")
    print(f"Columns: {columns}")
    print(f"Transformer object: {transformer}")
    print()

# Optional: check OneHotEncoder categories
for name, transformer, columns in preprocessor.transformers_:
    if hasattr(transformer, 'categories_'):
        print(f"{name} categories: {transformer.categories_}")
