import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from flask import Flask, request, json, jsonify
import os

app = Flask(__name__)

# 전처리 함수
def clean_ingredient_text(text):
    if not isinstance(text, str):
        return []
    items = re.split(r',\s*', text)
    names = []
    for item in items:
        parts = item.split('|')
        if len(parts) == 2:
            name = parts[0].strip()
            if name:
                names.append(name)
    return list(set(names))

# 데이터 로드 및 모델 학습
try:
    file_path = "recipes_dataset_revise_final.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} file not found.")
        df = pd.DataFrame()
    else:
        df = pd.read_csv(file_path, encoding="utf-8")
        df["parsed_ingredients"] = df["RCP_PARTS_DTLS"].apply(lambda x: clean_ingredient_text(str(x)))
        df["ingredient_text"] = df["parsed_ingredients"].apply(lambda x: ' '.join(x))

        vectorizer = TfidfVectorizer()
        ingredient_vectors = vectorizer.fit_transform(df["ingredient_text"])

        k = 10
        model = KMeans(n_clusters=k, random_state=42)
        df["cluster"] = model.fit_predict(ingredient_vectors)
        
        df["RCP_SEQ_STR"] = df["RCP_SEQ"].astype(str)

        print("Recipe data and model loaded successfully!")

except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    df = pd.DataFrame()

# 첫 번째 엔드포인트: 추천 목록 (간략 정보) 반환
@app.route('/recommend', methods=['POST'])
def recommend_recipes_list():
    if df.empty:
        return jsonify({"error": "Data not loaded. Please check recipes_dataset_revise_final.csv."}), 500

    try:
        data = request.get_json()
        
        if isinstance(data, list) and len(data) > 0:
            user_input = data[0].get('ingredients', '')
        else:
            user_input = data.get('ingredients', '')

        user_ingredients = [x.strip() for x in user_input.split(",") if x.strip()]

        if not user_ingredients:
            return jsonify({"message": "Please provide ingredients."}), 400

        user_text = ' '.join(user_ingredients)
        user_vector = vectorizer.transform([user_text])
        user_cluster = model.predict(user_vector)[0]

        cluster_df = df[df["cluster"] == user_cluster].copy()

        def get_match_count(recipe_ingredients, user_ingredients):
            count = 0
            for ui in user_ingredients:
                for ri in recipe_ingredients:
                    if ui in ri or ri in ui:
                        count += 1
                        break
            return count

        cluster_df["match_count"] = cluster_df["parsed_ingredients"].apply(
            lambda recipe: get_match_count(recipe, user_ingredients)
        )
        matched_df = cluster_df[cluster_df["match_count"] > 0].sort_values(
            by="match_count", ascending=False
        ).reset_index(drop=True)

        recommendations = []
        if not matched_df.empty:
            for _, row in matched_df.head(10).iterrows():
                recommendations.append({
                    "recipeId": str(row["RCP_SEQ"]), 
                    "recipeName": row["RCP_NM"],
                    "matchCount": int(row["match_count"]),
                    "dishType": row.get("RCP_PAT2", ""),
                    "hashtag": row.get("HASH_TAG", "")
                })
        
        json_response_body = json.dumps(recommendations, ensure_ascii=False, indent=4)
        response = app.response_class(
            response=json_response_body,
            status=200,
            mimetype='application/json; charset=utf-8'
        )
        return response
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Internal server error."}), 500

# 두 번째 엔드포인트: 상세 정보 반환
@app.route('/details/<string:recipe_id>', methods=['GET'])
def get_recipe_details(recipe_id):
    if df.empty:
        return jsonify({"error": "Data not loaded."}), 500

    recipe = df[df["RCP_SEQ_STR"] == recipe_id]
    if recipe.empty:
        return jsonify({"error": "Recipe not found."}), 404
    
    row = recipe.iloc[0]
    
    # 요리 방법에서 'nan' 값을 제거
    cooking_steps = []
    for i in range(1, 21):
        step = str(row.get(f"MANUAL{str(i).zfill(2)}", "")).strip()
        if step and step.lower() != 'nan':
            cooking_steps.append(re.sub(r'\s+', ' ', step))

    # 재료 리스트 변환
    ingredients_list = [
        {"name": part.split('|')[0].strip(), "amount": part.split('|')[1].strip()}
        for part in str(row["RCP_PARTS_DTLS"]).split(',')
        if '|' in part
    ]

    details = {
        "recipeName": row.get("RCP_NM", ""),
        "ingredients": ingredients_list,
        "cookingSteps": cooking_steps,
        "nutrition": {
            "energy": row.get("INFO_ENG", ""),         # kcal
            "carbohydrate": row.get("INFO_CAR", ""),
            "protein": row.get("INFO_PRO", ""),
            "fat": row.get("INFO_FAT", ""),
            "sodium": row.get("INFO_NA", "")
        },
        "tip": row.get("RCP_NA_TIP", "").strip()
    }
    
    json_response_body = json.dumps(details, ensure_ascii=False, indent=4)
    response = app.response_class(
        response=json_response_body,
        status=200,
        mimetype='application/json; charset=utf-8'
    )
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
