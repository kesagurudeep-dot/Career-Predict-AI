from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
company_encoder = joblib.load("company_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=["POST"])
def predict():

    branch = request.form["branch"]
    cgpa = float(request.form["cgpa"])
    python = 1 if request.form["python"] == "Yes" else 0
    java = 1 if request.form["java"] == "Yes" else 0
    sql = 1 if request.form["sql"] == "Yes" else 0
    projects = int(request.form["projects"])
    internship = 1 if request.form["internship"] == "Yes" else 0
    certifications = int(request.form["certifications"])

    branch_map = {
        "CSE": 0,
        "ECE": 1,
        "EEE": 2,
        "MECH": 3,
        "CIVIL": 4,
        "IT": 5
    }

    branch = branch.upper()
    branch_encoded = branch_map.get(branch, 0)

    features = [[
        branch_encoded,
        cgpa,
        python,
        java,
        sql,
        projects,
        internship,
        certifications
    ]]

    prediction = model.predict(features)[0]
    company = company_encoder.inverse_transform([prediction])[0]

    score = 0

    if cgpa >= 8.5:
        score += 30
    elif cgpa >= 7:
        score += 20
    else:
        score += 10

    if python:
        score += 10

    if java:
        score += 10

    if sql:
        score += 10

    score += min(projects * 5, 20)
    score += 10 if internship else 0
    score += min(certifications * 2, 10)

    recommendations = []

    if cgpa < 8:
        recommendations.append("Improve your CGPA.")

    if not python:
        recommendations.append("Learn Python programming.")

    if not java:
        recommendations.append("Improve your Java skills.")

    if not sql:
        recommendations.append("Learn SQL and database concepts.")

    if projects < 3:
        recommendations.append("Complete more academic or personal projects.")

    if not internship:
        recommendations.append("Gain internship experience.")

    if certifications < 2:
        recommendations.append("Earn more technical certifications.")

    return render_template("index.html", prediction=company, score=score, recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)