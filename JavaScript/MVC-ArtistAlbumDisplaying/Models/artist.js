const artists = 
[
    { id: 1, name: "Travis Scott", genre: "Hip Hop" },
    { id: 2, name: "Michael Jackson", genre: "Pop" },
];

module.exports = 
{
    findById: (id) => artists.find(artist => artist.id === id), //returns the artist object matching the id
};
