#include<map>
#include<cstdio>
#include<iostream>
#include<set>
#include<algorithm>
#include<cstdlib>
#include<tuple>
#include<string>
#include<stdlib.h>
#include"mcqd.h"
#include <fstream>
using namespace std;


int main(){
	//might want to redo to file opener.
	char file_path[] = "./dcts/dfa_12_try_7.dct";
	freopen(file_path,"r",stdin);
	freopen("output.txt","w",stdout);
	int train_size,alsize;
	scanf("%d %d",&train_size,&alsize);
	string train_string[train_size];
	int train_label[train_size];
	for(int i = 0; i<train_size;i++ ){
		int label, string_length;
		scanf("%d %d",&label,&string_length);
		string s;
		int count =0;
		while(count<string_length){
			int d;
			scanf(" %d",&d);
			s=s+to_string(d);
			count++;
			
		}
		train_string[i] = s;
		train_label[i]=label;
	}
	//enumerate fixes. generate the prefix and suffix sets
	set<string> prefixes, suffixes;
	string prefix, suffix;
	for(int i=0;i<train_size; i++){
		for(int j=0;j< train_string[i].length()+1; j++){
			prefix=train_string[i].substr(0,j);
			suffix=train_string[i].substr(j,train_string[i].length());
			prefixes.insert(prefix);
			suffixes.insert(suffix);
		}
	}
	
	// set dictionaries
	map<string,int> prefixes_map, suffixes_map;
	int count=0;
	for(set<string>::iterator i=prefixes.begin();i!=prefixes.end(); i++){
		prefixes_map[*i]=count++;
	}
	count=0;
	for(set<string>::iterator i=suffixes.begin();i!=suffixes.end(); i++){
		suffixes_map[*i]=count++;
	}
	/*	
	for(set<string>::iterator i=prefixes.begin();i!=prefixes.end(); i++){
		cout<<prefixes_map[*i]<<"   "<<*i<<"   \n";
	}
	cout<<"------\n";
	for(set<string>::iterator i=suffixes.begin();i!=suffixes.end(); i++){
		cout<<suffixes_map[*i]<<"   "<<*i<<"   \n";
	}
	*/
	map<int,map<int,set<int>>> dmap;

	for(int i=0;i<train_size; i++){
		for(int j=0;j< train_string[i].length()+1; j++){
			prefix=train_string[i].substr(0,j);
			suffix=train_string[i].substr(j,train_string[i].length());
			int pi = prefixes_map[prefix];
			int si = suffixes_map[suffix];
			
			dmap[si][train_label[i]].insert(pi);
			}
	}
	/*
	for(set<int>::iterator i=dmap[1][1].begin();i!=dmap[1][1].end(); i++){
		
		cout<<"hi   "<<*i<<"   \n";
	}
	*/
	/* max cliique alg
	bool **conn;
	conn = new bool*[prefixes.size()];
   	for (int i = 0; i < prefixes.size(); i++ ) {
      conn[i] = new bool[prefixes.size()];
   	}
	*/
	int edgeNum=0;
	bool conn[prefixes.size()][prefixes.size()];
	
	for(map<int,map<int,set<int>>>::iterator i=dmap.begin();i!=dmap.end();i++){
		map<int,set<int>> sec = i->second;
		if(sec.find(0)!=sec.end() && sec.find(1) != sec.end()){
			for(set<int>::iterator j=sec[0].begin(); j!=sec[0].end(); j++){
				for(set<int>::iterator k=sec[1].begin();k!=sec[1].end();k++){
					conn[*j][*k]=true;			
					conn[*k][*j]=true;
					edgeNum++;
				}
			}	
		}
	}
	
	ofstream dimacs;
  	dimacs.open ("search_dimacs.col");
	dimacs<<"p edge "<<prefixes.size()<<" "<<edgeNum<<"\n";
	for(map<int,map<int,set<int>>>::iterator i=dmap.begin();i!=dmap.end();i++){
		map<int,set<int>> sec = i->second;
		if(sec.find(0)!=sec.end() && sec.find(1) != sec.end()){
			for(set<int>::iterator j=sec[0].begin(); j!=sec[0].end(); j++){
				for(set<int>::iterator k=sec[1].begin();k!=sec[1].end();k++){
					dimacs<<"e "<<*j+1<<" "<<*k+1<<"\n";
				}
			}	
		}
	}
	
	int result = system("./../fastColor/fastColor -f search_dimacs.col -t 0");	
	freopen("output.txt","r",stdin);
	
	int read=0;
	char buffer[100];
	scanf("%*[^\n]\n");
	scanf("%*[^\n]\n");
	float f;
	char a[10];
	scanf("%*f %*s %*s %*s %d",&result); 
	
	//somehow find a max clique alg
	/* max clique alg
	int *qmax;
	int qsize;
	Maxclique m(conn,prefixes.size());
	m.mcq(qmax,qsize);
	cout<<qsize;
	*/

}

	

	
