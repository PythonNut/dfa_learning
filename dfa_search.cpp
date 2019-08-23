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
#include <bits/stdc++.h> 
using namespace std;

bool cmp(string a, string b) {
   	if(a.size()!=b.size())
		return a.size()<b.size();
	else{
		char aa[a.size()];
		char bb[b.size()];
		strcpy(aa,a.c_str());
		strcpy(bb,b.c_str());
		return lexicographical_compare(aa,aa+a.size(),bb,bb+b.size());
	}
}	
//access conflict by *((conflict+i*n)+j)
bool search( set<string, decltype(&cmp) > prefixes, map<string,int> prefixes_map,int dim, int *conflict ,int color){
	cout<<"hi";
	for(set<string>::iterator i=prefixes.begin();i!=prefixes.end();i++)
		cout<<*i<<"\n";
	if(color<11)
		return false;
	else
		return true;
}
int main(){
	//might want to redo to file opener.
	
	//read inputs from file. might want to optimize to take prefix set and suffix set in here.
	char file_path[] = "./dcts/dfa_12_try_7.dct";
	freopen(file_path,"r",stdin);
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
	set<string, decltype(&cmp)> prefixes(&cmp);
	set<string> suffixes;
	string prefix, suffix;
	for(int i=0;i<train_size; i++){
		for(int j=0;j< train_string[i].length()+1; j++){
			prefix=train_string[i].substr(0,j);
			suffix=train_string[i].substr(j,train_string[i].length());
			prefixes.insert(prefix);
			suffixes.insert(suffix);
		}
	}
	
	// build dictionaries
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
	//printing
	for(set<string>::iterator i=prefixes.begin();i!=prefixes.end(); i++){
		cout<<prefixes_map[*i]<<"   "<<*i<<"   \n";
	}
	cout<<"------\n";
	for(set<string>::iterator i=suffixes.begin();i!=suffixes.end(); i++){
		cout<<suffixes_map[*i]<<"   "<<*i<<"   \n";
	}
	*/

	// build distinguishability map
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
	
	/* printing
	for(set<int>::iterator i=dmap[1][1].begin();i!=dmap[1][1].end(); i++){
		
		cout<<"hi   "<<*i<<"   \n";
	}
	*/
	/*max cliique alg
	bool **conn;
	conn = new bool*[prefixes.size()];
   	for (int i = 0; i < prefixes.size(); i++ ) {
      conn[i] = new bool[prefixes.size()];
   	}
	*/
	
	//conflict graph
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

	//building conflict graph for coloring
	freopen("output.txt","w",stdout);
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
	system("./../fastColor/fastColor -f search_dimacs.col -t 0");	
	freopen("output.txt","r",stdin);
	int color;
	scanf("%*[^\n]\n");
	scanf("%*[^\n]\n");
	scanf("%*f %*s %*s %*s %d",&color); 
	cout<<color<<"\n";
	

	/* max clique alg
	int *qmax;
	int qsize;
	Maxclique m(conn,prefixes.size());
	m.mcq(qmax,qsize);
	cout<<qsize;
	*/

	// searching
	bool fail = true;
	while(fail){
		fail = search( prefixes,prefixes_map, prefixes.size(),(int *)conn,color);
		color++;
	}
	cout<<"end";
}


	

	
