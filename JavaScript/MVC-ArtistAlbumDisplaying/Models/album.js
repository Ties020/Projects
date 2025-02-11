const albums = 
[
    { id: 1, artistId: 1, title: "Astroworld", year: 2018 },
    { id: 2, artistId: 1, title: "Utopia", year: 2023 },
    { id: 3, artistId: 2, title: "Dangerous", year: 1991 },
];

module.exports = 
{
    findByArtistId: (artistId) => albums.filter(album => album.artistId === artistId) //returns albums where the artistId in the albums array matches the one that was passed as an argument
};
