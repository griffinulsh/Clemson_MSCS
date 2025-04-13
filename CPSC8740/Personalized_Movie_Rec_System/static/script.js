let ratings = [];

fetch('/get_movies')
.then(response => response.json())
.then(movies => {
  const container = document.getElementById('movies');
  container.innerHTML = ""; // Clear any existing content

  movies.forEach(movie => {
    container.innerHTML += `
      <div>
        <strong>${movie.title}</strong> (${movie.genre})
        <input type="number" min="1" max="5" id="${movie.title}" placeholder="Rate 1-5">
      </div>`;
  });
});

function submitRatings() {
  ratings = [];
  document.querySelectorAll('input').forEach(input => {
    if (input.value) {
      ratings.push({ title: input.id, rating: parseFloat(input.value) });
    }
  });

  fetch('/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ratings })
  })
  .then(response => response.json())
  .then(recommendations => {
    const recDiv = document.getElementById('recommendations');
    recDiv.innerHTML = "";
    recommendations.forEach(r => {
      recDiv.innerHTML += `<li>${r}</li>`;
    });
  });
}
