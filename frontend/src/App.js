import React, { useState } from 'react';

function App() {
  const [formData, setFormData] = useState({
    Pregnancies: 1,
    Glucose: 120,
    BloodPressure: 70,
    SkinThickness: 20,
    Insulin: 85,
    BMI: 25.5,
    DiabetesPedigreeFunction: 0.5,
    Age: 30
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: parseFloat(e.target.value) || 0
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      setResult(data);
    } catch (err) {
      alert("Error connecting to FastAPI backend. Ensure server.py is running!");
    }
    setLoading(false);
  };

  const handleDownloadPDF = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/generate-pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = "Clinical_Risk_Report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      alert("Error downloading PDF report.");
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-6 flex flex-col items-center">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-teal-400 bg-clip-text text-transparent">
          GlycoTrace
        </h1>
        <p className="text-slate-400 text-sm mt-1">Explainable AI Diagnostic & Reporting System</p>
      </header>

      <div className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Input Form Card */}
        <div className="bg-slate-800/50 backdrop-blur-md p-6 rounded-2xl border border-slate-700 shadow-xl">
          <h2 className="text-xl font-semibold mb-4 text-blue-400">Patient Clinical Vitals</h2>
          <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4">
            {Object.keys(formData).map((key) => (
              <div key={key} className="flex flex-col">
                <label className="text-xs text-slate-400 uppercase font-medium mb-1">{key}</label>
                <input
                  type="number"
                  step="any"
                  name={key}
                  value={formData[key]}
                  onChange={handleChange}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
                  required
                />
              </div>
            ))}
            <div className="col-span-2 mt-2">
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2.5 rounded-lg transition shadow-lg shadow-blue-600/30"
              >
                {loading ? "Analyzing..." : "Run AI Diagnostic"}
              </button>
            </div>
          </form>
        </div>

        {/* Results & Report Card */}
        <div className="bg-slate-800/50 backdrop-blur-md p-6 rounded-2xl border border-slate-700 shadow-xl flex flex-col justify-between">
          <div>
            <h2 className="text-xl font-semibold mb-4 text-teal-400">Diagnostic & XAI Results</h2>
            {result ? (
              <div className="space-y-4">
                <div className={`p-4 rounded-xl border ${result.prediction === 1 ? 'bg-red-950/40 border-red-800 text-red-300' : 'bg-emerald-950/40 border-emerald-800 text-emerald-300'}`}>
                  <p className="text-xs uppercase font-bold tracking-wider">Assessment</p>
                  <p className="text-2xl font-black mt-1">{result.risk_label}</p>
                  <p className="text-sm mt-1">Risk Probability: <span className="font-bold">{result.probability}%</span></p>
                </div>

                <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-700">
  <p className="text-xs font-semibold text-slate-400 mb-2">Engineered Feature Highlights:</p>
  <ul className="text-xs space-y-1 text-slate-300">
    <li>Glucose-BMI Interaction: <span className="font-mono text-blue-400">{result?.engineered_features?.Glucose_BMI_interaction}</span></li>
    <li>Insulin-Glucose Ratio: <span className="font-mono text-blue-400">{result?.engineered_features?.Insulin_Glucose_ratio}</span></li>
    <li>Genetic Age Exposure: <span className="font-mono text-blue-400">{result?.engineered_features?.Genetic_Age_exposure}</span></li>
  </ul>
</div>
              </div>
            ) : (
              <div className="h-48 flex items-center justify-center text-slate-500 text-center border border-dashed border-slate-700 rounded-xl">
                Enter patient metrics and click "Run AI Diagnostic" to view analysis.
              </div>
            )}
          </div>

          {result && (
            <button
              onClick={handleDownloadPDF}
              className="mt-6 w-full bg-emerald-600 hover:bg-emerald-500 text-white font-medium py-2.5 rounded-lg transition shadow-lg shadow-emerald-600/30 flex items-center justify-center gap-2"
            >
              📥 Download Clinical PDF Report
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;