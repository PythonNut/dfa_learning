#include<map>
#include<cstdio>
#include<iostream>
#include<set>
#include<algorithm>
#include<cstdlib>
#include<tuple>
#include<string>
using namespace std;


int main(){
	//might want to redo to file opener.
	freopen("./dcts/dfa_12_try_1.dct","r",stdin);
	int train_size,alsize;
	scanf("%d %d",&train_size,&alsize);
	string train_string[train_size];
	int train_label[train_size];
	for(int i = 0; i<train_size;i++ ){
		int label;
		long string_length;
		scanf("%d %ld",&label,&string_length);
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
	int conflict[prefixes.size()][prefixes.size()]={0};
	for(map<int,map<int,set<int>>>::iterator i=dmap.begin();i!=dmap.end();i++){
		map<int,set<int>> sec = i->second;
		if(sec.find(0)!=sec.end() && sec.find(1) != sec.end()){
			for(set<int>::iterator j=sec[0].begin(); j!=sec[0].end(); j++){
				for(set<int>::iterator k=sec[1].begin();k!=sec[1].end();k++){
					conflict[*j][*k]=1;			
					conflict[*k][*j]=1;			
				}
			}	
		}
	}
		
	
			
}

	

	
