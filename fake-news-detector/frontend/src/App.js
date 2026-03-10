import NewsChecker from "./components/NewsChecker";

function App() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-100 via-sky-100 to-cyan-100 px-4 py-10">
      <section className="mx-auto mb-8 max-w-3xl text-center">
        <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 md:text-4xl">
          Fake News Detection System
        </h1>
        <p className="mt-3 text-slate-600">
          Analyze article text using a TF-IDF based machine learning model.
        </p>
      </section>
      <NewsChecker />
    </main>
  );
}

export default App;
