const artistModel = require('../Models/artist');

exports.showArtist = (req, res) => 
{
    const artistId = parseInt(req.params.id, 10);
    const artist = artistModel.findById(artistId);
    
    if (artist) res.render('artistView', { artist });
    else res.status(404).send('Artist not found');
    
};