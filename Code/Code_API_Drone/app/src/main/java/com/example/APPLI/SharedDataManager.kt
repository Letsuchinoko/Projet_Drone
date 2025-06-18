package com.example.APPLI

import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class SharedDataManager : ViewModel() {
    private val TAG = "SharedDataManager"
    
    private val _gpsData = MutableLiveData<String>()
    val gpsData: LiveData<String> = _gpsData

    private val _soundData = MutableLiveData<String>()
    val soundData: LiveData<String> = _soundData

    fun updateGpsData(data: String) {
        Log.d(TAG, "updateGpsData appelé avec: '$data'")
        _gpsData.postValue(data)
        Log.d(TAG, "Valeur GPS mise à jour dans LiveData")
    }

    fun updateSoundData(data: String) {
        Log.d(TAG, "updateSoundData appelé avec: '$data'")
        _soundData.postValue(data)
        Log.d(TAG, "Valeur sonore mise à jour dans LiveData")
    }
} 