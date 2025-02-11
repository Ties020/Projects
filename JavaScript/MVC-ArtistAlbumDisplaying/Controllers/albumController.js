const albumModel = require('../Models/album');

exports.showAlbumsByArtist = (req, res) => 
{
    const artistId = parseInt(req.params.artistId, 10);
    const albums = albumModel.findByArtistId(artistId);
    
    if (albums.length) res.render('albumView', { albums });
    else res.status(404).send('No albums found for this artist');
    
};
