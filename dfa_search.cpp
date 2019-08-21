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
	fropen("test.txt","r",stdin);
	int train_size,alsize;
	scanf("%d %d",&train_size,&alsize);
	string train_string[train_size];
	int train_label[train_size];
	for(int i = 0; i<train_size;i++ ){
		int label,string_length;
		scanf("%d %d",&label,&string_length);
		char str[string_length];
		int count =0;
		while(count<string_length){
			scanf("%c ",&str[count++]);
		}
		train_string[i] = str;
		train_label[i]=label;
	}
	/*
	//enumerate fixes. generate the prefix and suffix sets
	set<string> prefixes, suffixes;
	string prefix, suffix;
	for(int i=0;i<train_size; i++){
		for(int j=0;j< train_string[i].length(); j++){
			prefix=train_string[i].substr(0,j);
			suffix=train_string[i].substr(j,train_string[i].length());
			prefixes.insert(prefix);
			suffixes.insert(suffix);
		}
	}



	// set dictionaries
	map<string,int> prefixes_map, suffixes_map;
	for(int i=0;i<prefixes.size(); i++){
		prefixes_map[prefixes[i]]=i;
	}
	for(int i=0;i<suffixes.size(); i++){
		suffixes_map[suffixes[i]]=i;
	}
	
	map<int,map<int,set<int>>> dmap;

	for(int i=0;i<train_size; i++){
		for(int j=0;j< train_string[i].length(); j++){
			prefix=train_string[i].substr(0,j);
			suffix=train_string[i].substr(j,train_string[i].length());
			int pi = prefixes_map[prefix];
			int si = suffixes_map[suffix];
			
			dmap[si][train_label[i]].insert(pi);
			}
	}
	int conflict[prefixes.size()][prefixes.size()];
	for(int i=0;i<dmap.size();i++){
		map<int,set<string>> try = *(dmap.begin()+i);
		if(try.find(0)!=try.end() && try.find(1) != try.end()){
			for(int j=0; j<try[0].size(); j++){
				for(int k=0;k<try[1].size();k++){
					conflict[*(try[0].begin()+j)][*(try[1].being()+k]=1;			
					conflict[*(try[1].begin()+k)][*(try[0].being()+j]=1;			
				}
			}	
		}
	}
	
	*/
	
			
}

	

	
