package com.example.APPLI

import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

// Classe qui permet de partager et de gérer les données GPS et sonores entre différentes parties de l'application
class SharedDataManager : ViewModel() {
    // Tag utilisé pour les logs
    private val TAG = "SharedDataManager"
    
    // LiveData privée pour stocker les données GPS
    private val _gpsData = MutableLiveData<String>()
    // LiveData publique pour observer les données GPS
    val gpsData: LiveData<String> = _gpsData

    // LiveData privée pour stocker les données sonores
    private val _soundData = MutableLiveData<String>()
    // LiveData publique pour observer les données sonores 
    val soundData: LiveData<String> = _soundData

    // Fonction pour mettre à jour les données GPS
    fun updateGpsData(data: String) {
        Log.d(TAG, "updateGpsData appelé avec: '$data'") // Log pour le debug
        _gpsData.postValue(data) // Met à jour la valeur de la LiveData GPS
        Log.d(TAG, "Valeur GPS mise à jour dans LiveData") // Log pour le debug
    }

    // Fonction pour mettre à jour les données sonores
    fun updateSoundData(data: String) {
        Log.d(TAG, "updateSoundData appelé avec: '$data'") // Log pour le debug
        _soundData.postValue(data) // Met à jour la valeur de la LiveData sonore
        Log.d(TAG, "Valeur sonore mise à jour dans LiveData") // Log pour le debug
    }
} 