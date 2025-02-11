const express = require('express');
const artistController = require('./Controllers/artistController');
const albumController = require('./Controllers/albumController');
const path = require('path');

const app = express();
const PORT = 3001;

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'Views'));

//this front controller handles all requests
app.use((req, res, next) => 
{
    console.log('Incoming request:', req.url);
    next(); //pass control to the next handler, one of the two which are below
});

app.get('/artist/:id', artistController.showArtist); //call artistController to handle get request when wanting to display artist info
app.get('/artist/:artistId/albums', albumController.showAlbumsByArtist); //call albumController to handle get request when wanting to display artist's albums

app.listen(PORT, () => {
    console.log(`Artist and Album app running on http://localhost:${PORT}`);
});
