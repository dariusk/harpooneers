var request = require('request');
var cheerio = require('cheerio');
var _ = require('underscore');
var Twit = require('twit');
var T = new Twit(require('./config.js'));
var wordfilter = require('wordfilter');
var ent = require('ent');
var rita = require('rita');
var lexicon = new rita.RiLexicon();
var r = rita.RiTa;
var inquirer = require("inquirer");
var gutencorpus = require('gutencorpus');
var exec = require('child_process').exec;

const METHOD = doc2vec;

Array.prototype.pick = function() {
  return this[Math.floor(Math.random()*this.length)];
};

Array.prototype.pickRemove = function() {
  var index = Math.floor(Math.random()*this.length);
  return this.splice(index,1)[0];
};

function randomWord() {
  return new Promise((resolve, reject) => resolve(lexicon.randomWord())).catch((e) => console.log(e));
}

function guten(word) {
  return new Promise((resolve, reject) => {
    word = word || lexicon.randomWord();
    gutencorpus.search(word, {caseSensitive: true}).done(result => resolve(result.pick()));
  });
}

function doc2vec(num) {
  return new Promise((resolve, reject) => {
    child = exec('python ../../doc2vec/gen-non.py ' + num,
      function (error, stdout, stderr) {
        //console.log('stdout: ' + stdout);
        //console.log('stderr: ' + stderr);
        if (error !== null) {
          console.log('exec error: ' + error);
        }
        else {
          resolve(stdout);
        }
    });
  });
}

function generate(func, num) {
  return new Promise((resolve, reject) => {
    var result = {
      text: '',
      choices: []
    };
    func(num).then(text => {
      result.choices[0] = text.match(/(.*)\n/)[1];
      result.text += `@@0 ${text}`;
      for (var i=1;i<10;i++) {
        result.text = result.text.replace('\n','@@'+i+' ');
        var reg = new RegExp('@@'+(i)+' (.*)\n');
        result.choices[i] = result.text.match(reg)[1];
      }
      result.text = result.text.replace(/@@/g,'\n');
      resolve(result);
    });

      /*
      var url = 'someUrl';
      request(url, (error, response, body) => {
        if (!error && response.statusCode == 200) {
          var result = '';
          var $ = cheerio.load(body);
          // parse stuff and resolve
          resolve(result);
        }
        else {
          reject();
        }
      });
      */
      
  }).catch((e) => console.log(e));
}

function search(term) {
  return new Promise((resolve, reject) => {
    console.log(`searching ${term}`);
    T.get('search/tweets', { q: term, count: 100 }, (err, reply) => {
      if (err) {
        throw new Error(`Search error: ${err}`);
      }
      else {
        var tweets = reply.statuses;
        tweets = _.chain(tweets)
          // decode weird characters
          .map(el => el.retweeted_status ? ent.decode(el.retweeted_status.text) : ent.decode(el.text))
          // throw out quotes and links and replies
          .reject(el => el.indexOf('http') > -1 || el.indexOf('@') > -1 || el.indexOf('"') > -1)
          .uniq()
          .value();
        resolve(tweets);
      }
    });
  }).catch((e) => console.log(e));
}

function question(num) {
  return new Promise((resolve, reject) => {
    generate(METHOD, num).then(result => {
      result.choices = result.choices.filter(choice => +choice.match(/{{(\d+)}}/)[1] > 5414)
      result.choices = result.choices.filter(choice => !choice.match(/nigg/))
      if (result.choices.length > 0) {
        resolve(result.choices[0]);
      }
      else {
        resolve('\n*****\n {{123}}');
      }
    });
  }).catch((e) => console.log(e));
}

var total = 5414;
var count = 0;
var novel = '';

function nextQuestion(num) {
  question(num).then(result => {
    count++;
    var snip = result.replace(/ {.*/,'');
    var newline = ' ';
    if (snip.trim()[0] === `'` || snip.trim()[0] === `"` || Math.random() < 0.2) {
      newline = '\n';
    }
    novel += `${newline}${snip}`;
    console.log(`#${count}: ${snip}`);
    var fs = require('fs');
    fs.appendFile(__dirname+'/novel-most-non.txt', `${newline}${snip}`, err => {
      if(err) {
        return console.log(err);
      }
      console.log('The file was saved!');
    }); 
    var id = +result.match(/{{(\d+)}}/)[1];
    if (count < total) {
      if (Math.random() < 1.65) {
        console.log('stick with this book...');
        nextQuestion(num+1);
      }
      else {
        console.log('jump to the chosen passage...');
        nextQuestion(id+1);
      }
    }
    else {
      console.log('novel complete!');
      console.log(novel);
    }
  });
}

nextQuestion(180);
