package com.gunadarma.heartratearrhythmiachecker.model;

import androidx.lifecycle.ViewModel;

import com.gunadarma.heartratearrhythmiachecker.service.DataRecordServiceImpl;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class SharedViewModel extends ViewModel {
    private DataRecordServiceImpl dataRecordService;
}